# ══════════════════════════════════════════════════════════════════════════════
# TIGS Analysis — TCGA LUAD & LUSC
# ══════════════════════════════════════════════════════════════════════════════

# ── libraries ─────────────────────────────────────────────────────────────────
library(tidyverse)
library(data.table)
library(survival)
library(survminer)
library(pROC)
library(GSVA)
library(ggstatsplot)
library(corrplot)
library(forestplot)
library(ggpubr)
library(pheatmap)
library(scales)
library(UCSCXenaTools)
library(TCGAmutations)
library(maftools)

setwd('/home/daniel/Desktop/R_data_prep')
dir.create("results",               showWarnings = FALSE)
dir.create("UCSC_Xena/TCGA/Survival", showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DOWNLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

xe <- XenaGenerate(subset = XenaHostNames == "tcgaHub")

# helper: find datasets matching pattern, filter, download
xena_download <- function(pattern, destdir) {
  all_ds   <- xe %>% XenaFilter(filterDatasets = pattern) %>% datasets()
  lung_ds  <- all_ds[grepl("LUAD|LUSC", all_ds)]
  message("Downloading: "); print(lung_ds)
  xe %>%
    XenaFilter(filterDatasets = paste(lung_ds, collapse = "|")) %>%
    XenaQuery() %>%
    XenaDownload(destdir = destdir, trans_slash = TRUE, force = TRUE)
}

xena_download("clinical",        "UCSC_Xena/TCGA/Clinical")
xena_download("HiSeqV2_PANCAN$", "UCSC_Xena/TCGA/RNAseq_Pancan")

# pan-cancer survival table
download.file(
  url      = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp",
  destfile = "UCSC_Xena/TCGA/Survival/pancan_survival.txt",
  mode     = "wb"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLEAN CLINICAL DATA
# ══════════════════════════════════════════════════════════════════════════════

fix_clinical <- function(path, project) {
  df <- data.table::fread(path)
  
  get_col <- function(col) {
    if (col %in% colnames(df)) df[[col]] else NA
  }
  
  tibble(
    Project              = project,
    Tumor_Sample_Barcode = df$sampleID,
    Age                  = as.numeric(get_col("age_at_initial_pathologic_diagnosis")),
    Gender               = get_col("gender"),
    Smoking_history      = get_col("tobacco_smoking_history"),
    Smoking_indicator    = get_col("tobacco_smoking_history_indicator"),
    sample_type          = get_col("sample_type"),
    pathologic_M         = get_col("pathologic_M"),
    pathologic_N         = get_col("pathologic_N"),
    pathologic_T         = get_col("pathologic_T"),
    pathologic_stage     = get_col("pathologic_stage"),
    Tumor_stage = case_when(
      get_col("pathologic_stage") %in% c("Stage I","Stage IA","Stage IB")                ~ "I",
      get_col("pathologic_stage") %in% c("Stage II","Stage IIA","Stage IIB","Stage IIC") ~ "II",
      get_col("pathologic_stage") %in% c("Stage IIIA","Stage IIIB","Stage IIIC")         ~ "III",
      get_col("pathologic_stage") %in% c("Stage IV","Stage IVA","Stage IVB","Stage IVC") ~ "IV",
      TRUE ~ NA_character_
    )
  ) %>%
    mutate(
      Gender      = factor(case_when(
        Gender == "FEMALE" ~ "Female",
        Gender == "MALE"   ~ "Male",
        TRUE               ~ NA_character_
      ), levels = c("Male","Female")),
      Tumor_stage = factor(Tumor_stage, levels = c("I","II","III","IV"))
    )
}

# load and merge survival from pan-cancer table
lung_surv <- fread("UCSC_Xena/TCGA/Survival/pancan_survival.txt") %>%
  filter(`cancer type abbreviation` %in% c("LUAD","LUSC")) %>%
  select(sample, OS, OS.time) %>%
  rename(Tumor_Sample_Barcode = sample) %>%
  mutate(Event = as.numeric(OS),
         Time  = as.numeric(OS.time)) %>%
  select(Tumor_Sample_Barcode, Event, Time)

TCGA_Clinical.tidy <- bind_rows(
  fix_clinical("UCSC_Xena/TCGA/Clinical/TCGA.LUAD.sampleMap__LUAD_clinicalMatrix", "LUAD"),
  fix_clinical("UCSC_Xena/TCGA/Clinical/TCGA.LUSC.sampleMap__LUSC_clinicalMatrix",    "LUSC")
) %>%
  filter(sample_type == "Primary Tumor") %>%
  left_join(lung_surv, by = "Tumor_Sample_Barcode")

message("Clinical — Event:"); print(table(TCGA_Clinical.tidy$Event))
message("Clinical — Time:");  print(summary(TCGA_Clinical.tidy$Time))
message("Clinical — Rows: ",  nrow(TCGA_Clinical.tidy))

save(TCGA_Clinical.tidy, file = "results/TCGA_tidy_Clinical_LUNG.RData")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD & SAVE RNASEQ
# ══════════════════════════════════════════════════════════════════════════════

RNAseq_filelist <- dir("UCSC_Xena/TCGA/RNAseq_Pancan", full.names = TRUE)
RNAseq_filelist <- RNAseq_filelist[grepl("LUAD|LUSC", RNAseq_filelist)]

RNASeq_List        <- XenaPrepare(RNAseq_filelist)
names(RNASeq_List) <- sub("TCGA\\.(.*)\\.sampleMap.*", "\\1", names(RNASeq_List))
RNASeq_lung        <- purrr::reduce(RNASeq_List, full_join)

save(RNASeq_lung, file = "results/TCGA_RNASeq_LUNG.RData")
rm(RNASeq_List)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GENE LISTS
# ══════════════════════════════════════════════════════════════════════════════

APM_genes <- read_csv("Xdata/APM.csv", skip = 1) %>%
  transmute(Cell_type = "APM", Symbol = Gene_Name, inRNAseq = "YES")

immune_cellType <- read_csv("Xdata/Immune_Cell_type_List.csv", skip = 1) %>%
  filter(inRNAseq == "YES") %>%
  select(Cell_type, Symbol, inRNAseq)

merged_geneList <- bind_rows(immune_cellType, APM_genes)
save(merged_geneList, file = "results/merged_geneList.RData")

rm(APM_genes, immune_cellType)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GSVA SCORING
# ══════════════════════════════════════════════════════════════════════════════

load("results/merged_geneList.RData")
load("results/TCGA_RNASeq_LUNG.RData")

expr_mat           <- as.data.frame(RNASeq_lung)
rownames(expr_mat) <- expr_mat[, 1]
expr_mat           <- as.matrix(expr_mat[, -1])

gset_list <- split(merged_geneList$Symbol, merged_geneList$Cell_type)

gsva_res <- gsva(
  expr          = expr_mat,
  gset.idx.list = gset_list,
  method        = "gsva",
  kcdf          = "Gaussian",
  verbose       = TRUE
)

gsva_df <- as.data.frame(t(gsva_res)) %>%
  rownames_to_column(var = "tsb") %>%
  mutate(
    TIS = (`CD8 T cells` + `T helper cells` + `T cells` + `Tcm cells` +
             `Tem cells`   + `Th1 cells`      + `Th2 cells` + `Th17 cells` +
             `Treg cells`) / 9,
    IIS = (`CD8 T cells` + `T helper cells` + `T cells`  + `Tcm cells` +
             `Tem cells`   + `Th1 cells`      + `Th2 cells` + `Th17 cells` +
             `Treg cells`  + aDC + `B cells`  + `Cytotoxic cells` + DC +
             Eosinophils   + iDC + Macrophages + `Mast cells` + Neutrophils +
             `NK CD56bright cells` + `NK CD56dim cells` + `NK cells` + pDC) / 22
  )

save(gsva_df, file = "results/gsva_lung.RData")
rm(RNASeq_lung, expr_mat, gsva_res)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TMB
# ══════════════════════════════════════════════════════════════════════════════

get_tmb <- function(study) {
  maf       <- tcga_load(study = study, source = "MC3")
  silent    <- maf@maf.silent[, .N, .(Tumor_Sample_Barcode)]
  nonsilent <- getSampleSummary(maf)
  full_join(silent, nonsilent, by = "Tumor_Sample_Barcode") %>%
    mutate(
      TMB_NonsynVariants = total,
      TMB_Total          = ifelse(!is.na(N), N + total, total)
    ) %>%
    select(Tumor_Sample_Barcode, TMB_NonsynVariants, TMB_Total)
}

LUNG_TMB <- bind_rows(get_tmb("LUAD"), get_tmb("LUSC")) %>%
  mutate(tsb12 = substr(Tumor_Sample_Barcode, 1, 12))

save(LUNG_TMB, file = "results/LUNG_TMB.RData")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUILD FINAL DATASET
# ══════════════════════════════════════════════════════════════════════════════

load("results/TCGA_tidy_Clinical_LUNG.RData")
load("results/gsva_lung.RData")
load("results/LUNG_TMB.RData")

lung_all <- TCGA_Clinical.tidy %>%
  full_join(gsva_df, by = c("Tumor_Sample_Barcode" = "tsb")) %>%
  filter(sample_type == "Primary Tumor", !is.na(Tumor_stage)) %>%
  mutate(tsb12 = substr(Tumor_Sample_Barcode, 1, 12)) %>%
  left_join(LUNG_TMB %>% select(-Tumor_Sample_Barcode), by = "tsb12") %>%
  mutate(
    nAPM = (APM - min(APM, na.rm = TRUE)) /
      (max(APM, na.rm = TRUE) - min(APM, na.rm = TRUE)),
    nTMB = TMB_NonsynVariants / 38,
    TIGS = log(nTMB + 1) * nAPM
  )

message("lung_all — Event:"); print(table(lung_all$Event))
message("lung_all — Time:");  print(summary(lung_all$Time))
message("lung_all — Rows: ",  nrow(lung_all))

save(lung_all, file = "results/LUNG_ALL.RData")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

load("results/LUNG_ALL.RData")

df_os <- lung_all %>%
  filter(!is.na(Time), !is.na(Event), Time > 0, Event %in% c(0, 1))

message("df_os — Rows: ", nrow(df_os))
print(table(df_os$Project))

# ── univariable Cox per cohort ─────────────────────────────────────────────────
run_cox <- function(var) {
  df_os %>%
    filter(!is.na(.data[[var]])) %>%
    group_by(Project) %>%
    dplyr::do({
      dat <- .
      tryCatch({
        fit <- coxph(Surv(time = Time, event = Event) ~ .data[[var]], data = dat)
        s   <- summary(fit)
        tibble(
          Coef   = s$conf.int[1, 1],
          Lower  = s$conf.int[1, 3],
          Upper  = s$conf.int[1, 4],
          Pvalue = s$logtest[3],
          N      = s$n
        )
      }, error = function(e) {
        message("Skipping ", unique(dat$Project), ": ", e$message)
        tibble(Coef = NA_real_, Lower = NA_real_,
               Upper = NA_real_, Pvalue = NA_real_, N = NA_integer_)
      })
    }) %>%
    ungroup()
}

cox_APS  <- run_cox("nAPM")
cox_TMB  <- run_cox("nTMB")
cox_TIGS <- run_cox("TIGS")

print(cox_APS)
print(cox_TMB)
print(cox_TIGS)

# ── Kaplan-Meier plots ────────────────────────────────────────────────────────
km_plot <- function(var, project) {
  df_sub <- df_os %>%
    filter(Project == project, !is.na(.data[[var]])) %>%
    mutate(group = ifelse(.data[[var]] >= median(.data[[var]], na.rm = TRUE),
                          "High", "Low"))
  
  fit <- survfit(Surv(Time, Event) ~ group, data = df_sub)
  
  ggsurvplot(
    fit,
    data       = df_sub,
    pval       = TRUE,
    risk.table = TRUE,
    title      = paste(project, "—", var, "(median split)"),
    palette    = c("#D85A30","#378ADD"),
    xlab       = "Days",
    ylab       = "Overall survival probability"
  )
}

for (proj in c("LUAD","LUSC")) {
  print(km_plot("TIGS", proj))
  print(km_plot("nAPM", proj))
  print(km_plot("nTMB", proj))
}

# ── APS vs IIS correlation ────────────────────────────────────────────────────
for (proj in c("LUAD","LUSC")) {
  p <- ggscatterstats(
    data     = filter(df_os, Project == proj, !is.na(APM), !is.na(IIS)),
    x        = APM,
    y        = IIS,
    xlab     = "APM Score (APS)",
    ylab     = "IIS",
    title    = proj,
    type     = "spearman",
    messages = FALSE
  )
  print(p)
}

# ── distribution plots ────────────────────────────────────────────────────────
plot_dist <- function(var, ylab) {
  df_os %>%
    filter(!is.na(.data[[var]])) %>%
    ggboxplot(
      x        = "Project", y = var,
      color    = "Project", add = "jitter",
      add.params = list(size = 0.6),
      xlab     = "TCGA Project", ylab = ylab,
      legend   = "none"
    ) +
    geom_hline(yintercept = mean(df_os[[var]], na.rm = TRUE), linetype = 2) +
    scale_color_manual(values = c("LUAD" = "#378ADD", "LUSC" = "#D85A30"))
}

print(plot_dist("APM",  "APM Score (APS)"))
print(plot_dist("nTMB", "TMB (mut/Mb)"))
print(plot_dist("TIGS", "TIGS"))