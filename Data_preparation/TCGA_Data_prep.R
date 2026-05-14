library(TCGAbiolinks)
library(SummarizedExperiment)
library(edgeR)
library(limma)
library(dplyr)

setwd('/home/daniel/Desktop/R_prep_data1')



dir.create("data", showWarnings = FALSE)
dir.create("data/tcga_gdc", showWarnings = FALSE)

query_rna <- GDCquery(
  project       = c("TCGA-LUAD", "TCGA-LUSC"),
  data.category = "Transcriptome Profiling",
  data.type     = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  sample.type   = "Primary Tumor"
)

#Dowloading the data set

GDCdownload(query_rna,
            method = "api",
            files.per.chunk = 10,
            directory = "data/tcga_gdc")

projects <- c("TCGA-LUAD", "TCGA-LUSC")
se_list <- list()

for (proj in projects) {
  query <- GDCquery(
    project       = proj,
    data.category = "Transcriptome Profiling",
    data.type     = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    sample.type   = "Primary Tumor"
  )
  
  se <- GDCprepare(query, directory = "data/tcga_gdc")
  se_list[[proj]] <- se
}

# Function to extract and process expression data
process_project <- function(se, project_name) {
  
  counts <- assay(se)
  gene_info <- as.data.frame(rowData(se))
  
  dge <- DGEList(counts = counts)
  keep <- filterByExpr(dge)
  dge <- dge[keep, , keep.lib.sizes = FALSE]
  dge <- calcNormFactors(dge)
  logCPM <- cpm(dge, log = TRUE)
  
  # Map gene symbols
  gene_symbols <- gene_info$gene_name
  rownames(logCPM) <- gene_symbols[match(rownames(logCPM), rownames(gene_info))]
  logCPM <- logCPM[!is.na(rownames(logCPM)), ]
  
  # ============================================================
  # PRIORITY 1 FEATURES (your existing extraction)
  # ============================================================
  
  cd8 <- colMeans(logCPM[rownames(logCPM) %in% c("CD8A", "CD8B"), , drop = FALSE], na.rm = TRUE)
  cyto <- colMeans(logCPM[rownames(logCPM) %in% c("GZMA", "GZMB", "PRF1"), , drop = FALSE], na.rm = TRUE)
  pd1 <- logCPM["PDCD1", ]
  pdl1 <- logCPM["CD274", ]
  antigen <- colMeans(logCPM[rownames(logCPM) %in% c("HLA-A", "HLA-B", "HLA-C"), , drop = FALSE], na.rm = TRUE)
  
  # ============================================================
  # PRIORITY 2 FEATURES (NEW - slow killing + recruitment)
  # ============================================================
  
  # Slow killing gene
  faslg <- logCPM["FASLG", ]
  
  # Recruitment chemokines
  cxcl9 <- logCPM["CXCL9", ]
  cxcl10 <- logCPM["CXCL10", ]
  ccl5 <- logCPM["CCL5", ]
  
  # Mean chemokine signature
  chemokine_genes <- c("CXCL9", "CXCL10", "CCL5")
  chemokine_expr <- logCPM[rownames(logCPM) %in% chemokine_genes, , drop = FALSE]
  chemokine_signature <- colMeans(chemokine_expr, na.rm = TRUE)
  
  # ============================================================
  # COMBINE ALL FEATURES
  # ============================================================
  
  features <- data.frame(
    sample = colnames(logCPM),
    project = project_name,
    
    # Priority 1
    CD8 = as.numeric(cd8),
    Cytotoxicity = as.numeric(cyto),
    Antigenicity = as.numeric(antigen),
    PD1 = as.numeric(pd1),
    PDL1 = as.numeric(pdl1),
    
    # Priority 2 - individual genes
    FASLG = as.numeric(faslg),
    CXCL9 = as.numeric(cxcl9),
    CXCL10 = as.numeric(cxcl10),
    CCL5 = as.numeric(ccl5),
    
    # Priority 2 - combined signatures
    Chemokine_Signature = as.numeric(chemokine_signature),
    
    stringsAsFactors = FALSE
  )
  
  # Remove rows with NAs in critical columns
  features <- features %>%
    filter(!is.na(CD8) & !is.na(Cytotoxicity) & !is.na(Antigenicity) & 
             !is.na(PD1) & !is.na(PDL1))
  
  return(features)
}

# Process both projects
luad_features <- process_project(se_list[["TCGA-LUAD"]], "LUAD")
lusc_features <- process_project(se_list[["TCGA-LUSC"]], "LUSC")

# Combine
lung_features <- rbind(luad_features, lusc_features)

print(paste("Extracted features for", nrow(lung_features), "samples"))
print(paste("LUAD:", nrow(luad_features), "samples"))
print(paste("LUSC:", nrow(lusc_features), "samples"))

# Save combined features
write.csv(lung_features, "data/TCGA_Lung_All_Features.csv", row.names = FALSE)
write.csv(luad_features, "data/LUAD_features_extended.csv", row.names = FALSE)
write.csv(lusc_features, "data/LUSC_features_extended.csv", row.names = FALSE)

print("✓ Saved:")
print("  - data/TCGA_Lung_All_Features.csv (combined)")
print("  - data/LUAD_features_extended.csv")
print("  - data/LUSC_features_extended.csv")




get_tcga_clinical_from_api <- function(project_code, output_file = NULL) {
  
  safe_get <- function(x, default = NA) {
    if (is.null(x) || length(x) == 0) default else x[[1]]
  }
  
  print(paste("Querying", project_code, "from GDC API..."))
  
  url <- "https://api.gdc.cancer.gov/cases"
  
  query_params <- list(
    filters = toJSON(
      list(
        op = "in",
        content = list(
          field = "project.project_id",
          value = project_code
        )
      ),
      auto_unbox = TRUE
    ),
    fields = paste(
      "case_id",
      "submitter_id",
      "project.project_id",
      "demographic.gender",
      "demographic.vital_status",
      "demographic.days_to_death",
      "diagnoses.age_at_diagnosis",
      "diagnoses.ajcc_pathologic_stage",
      "diagnoses.ajcc_pathologic_t",        # <-- added
      "diagnoses.tumor_largest_dimension_diameter",
      "diagnoses.days_to_last_follow_up",
      sep = ","
    ),
    expand = "diagnoses,demographic",
    size = 10000,
    format = "JSON"
  )
  
  response <- GET(url, query = query_params)
  
  if (status_code(response) != 200) {
    stop(paste("GDC API error:", status_code(response)))
  }
  
  content <- fromJSON(content(response, "text"), simplifyVector = FALSE)
  cases <- content$data$hits
  
  print(paste("Retrieved", length(cases), "cases"))
  
  clinical_list <- list()
  
  for (i in seq_along(cases)) {
    case <- cases[[i]]
    
    patient_id   <- safe_get(case$case_id,          NA_character_)
    submitter_id <- safe_get(case$submitter_id,      NA_character_)
    project      <- safe_get(case$project$project_id, NA_character_)
    
    # --- Demographic (case level) ---
    demo         <- case$demographic
    gender        <- safe_get(demo$gender,        NA_character_)
    vital_status  <- safe_get(demo$vital_status,  NA_character_)
    days_to_death <- as.numeric(safe_get(demo$days_to_death, NA_real_))
    
    # --- Diagnosis level ---
    if (!is.null(case$diagnoses) && length(case$diagnoses) > 0) {
      diag <- case$diagnoses[[1]]
      
      age_raw <- safe_get(diag$age_at_diagnosis)
      age_at_diagnosis <- if (!is.na(age_raw)) as.numeric(age_raw) / 365.25 else NA_real_
      
      pathologic_stage <- safe_get(diag$ajcc_pathologic_stage, NA_character_)
      
      # T-stage for tumor size proxy
      t_stage <- safe_get(diag$ajcc_pathologic_t, NA_character_)
      
      # Replace the tumor_size_mm case_when block with this:
      tumor_size_mm <- case_when(
        # Specific subcategories first (order matters)
        grepl("^T1a", t_stage, ignore.case = TRUE) ~ 10,   # ≤1 cm
        grepl("^T1b", t_stage, ignore.case = TRUE) ~ 15,   # 1–2 cm
        grepl("^T1c", t_stage, ignore.case = TRUE) ~ 25,   # 2–3 cm
        grepl("^T1",  t_stage, ignore.case = TRUE) ~ 15,   # T1 unspecified → midpoint
        grepl("^T2a", t_stage, ignore.case = TRUE) ~ 35,   # 3–4 cm
        grepl("^T2b", t_stage, ignore.case = TRUE) ~ 45,   # 4–5 cm
        grepl("^T2",  t_stage, ignore.case = TRUE) ~ 40,   # T2 unspecified → midpoint
        grepl("^T3",  t_stage, ignore.case = TRUE) ~ 60,   # 5–7 cm
        grepl("^T4",  t_stage, ignore.case = TRUE) ~ 80,   # >7 cm
        TRUE ~ NA_real_
      )
      dtlf_raw <- safe_get(diag$days_to_last_follow_up)
      days_to_last_follow_up <- if (!is.na(dtlf_raw)) as.numeric(dtlf_raw) else NA_real_
      
    } else {
      age_at_diagnosis       <- NA_real_
      pathologic_stage       <- NA_character_
      t_stage                <- NA_character_
      tumor_size_mm          <- NA_real_
      days_to_last_follow_up <- NA_real_
    }
    
    survival_days   <- if (!is.na(days_to_death)) days_to_death else days_to_last_follow_up
    survival_months <- if (!is.na(survival_days)) survival_days / 30.44 else NA_real_
    
    clinical_list[[i]] <- data.frame(
      case_id                = patient_id,
      submitter_id           = submitter_id,
      project                = project,
      age_at_diagnosis_years = age_at_diagnosis,
      gender                 = gender,
      pathologic_stage       = pathologic_stage,
      t_stage                = t_stage,          # raw T-stage kept for reference
      tumor_size_mm          = tumor_size_mm,    # derived from T-stage
      vital_status           = vital_status,
      survival_months        = survival_months,
      stringsAsFactors       = FALSE
    )
  }
  
  clinical_df <- do.call(rbind, clinical_list)
  rownames(clinical_df) <- NULL
  
  if (!is.null(output_file)) {
    write.csv(clinical_df, output_file, row.names = FALSE)
    print(paste("✓ Saved:", output_file))
  }
  
  return(clinical_df)
}


library(httr)
library(jsonlite)
library(dplyr)


# Get LUAD clinical
clinical_luad <- get_tcga_clinical_from_api(
  "TCGA-LUAD",
  output_file = "data/TCGA_LUAD_Clinical_FromAPI.csv"
)

# Get LUSC clinical
clinical_lusc <- get_tcga_clinical_from_api(
  "TCGA-LUSC",
  output_file = "data/TCGA_LUSC_Clinical_FromAPI.csv"
)

# Combine
clinical_all <- rbind(clinical_luad, clinical_lusc)

print(paste("✓ Total clinical records:", nrow(clinical_all)))
print(paste("  LUAD:", nrow(clinical_luad)))
print(paste("  LUSC:", nrow(clinical_lusc)))


print("Cleaning clinical data...")

clinical_all <- clinical_all %>%
  mutate(
    pathologic_stage = case_when(
      grepl("stage i[^iv]|stage 1|^i$", pathologic_stage, ignore.case = TRUE) ~ "Stage I",
      grepl("stage ii[^i]|stage 2|^ii$", pathologic_stage, ignore.case = TRUE) ~ "Stage II",
      grepl("stage iii[^i]|stage 3|^iii$", pathologic_stage, ignore.case = TRUE) ~ "Stage III",
      grepl("stage iv|stage 4|^iv$", pathologic_stage, ignore.case = TRUE)      ~ "Stage IV",
      TRUE ~ NA_character_
    ),
    vital_status = tolower(vital_status),
    sample_id    = submitter_id
  ) %>%
  select(
    sample_id, case_id, project,
    age_at_diagnosis_years, gender,
    pathologic_stage, t_stage, tumor_size_mm,
    vital_status, survival_months
  ) %>%
  distinct()

write.csv(clinical_all, "data/TCGA_Lung_Clinical_Cleaned.csv", row.names = FALSE)
print("✓ Saved cleaned clinical data")

print(paste("Total records:",    nrow(clinical_all)))
print(paste("With age:",         sum(!is.na(clinical_all$age_at_diagnosis_years))))
print(paste("With stage:",       sum(!is.na(clinical_all$pathologic_stage))))
print(paste("With t_stage:",     sum(!is.na(clinical_all$t_stage))))
print(paste("With tumor size:",  sum(!is.na(clinical_all$tumor_size_mm))))
print(paste("With survival:",    sum(!is.na(clinical_all$survival_months))))


