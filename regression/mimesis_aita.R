# --- Libraries ----------------------------------------------------------------
library(estimatr)
library(sandwich)
library(lmtest)
library(stargazer)
library(xtable)
library(arm)
library(grDevices)
library(dplyr)
library(car)
library(fixest)
library(modelsummary)

# --- Config -------------------------------------------------------------------
setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

# --- Function -----------------------------------------------------------------
run_mimesis_models <- function(main_var = "with_context", table_tag = "with_context", model_name = "claude-sonnet-4-20250514") {
  
  # Load & subset
  x <- read.csv(file = "mimesis.csv", stringsAsFactors = FALSE, header = TRUE)
  x <- subset(x, model==model_name)
  x <- subset(x, task=="aita")
  
  aita_terms <- grep("^aita_", names(x), value = TRUE)
  
  left_out_fe <- "aita_1gyx5dc"
  fe_terms <- setdiff(aita_terms, left_out_fe)
  
  # Build formulas dynamically --------------------------------------------------
  # FE part (exclude the left-out FE so it's the reference category)
  fe_part <- paste(fe_terms, collapse = " + ")
  
  # Interactions with the dynamic main var (using updated column names)
  int_liberal         <- paste0(main_var, ":is_liberal")
  int_not_liberal     <- paste0(main_var, ":is_not_liberal")
  int_man             <- paste0(main_var, ":is_man")
  int_not_man         <- paste0(main_var, ":is_not_man")
  
  int_lib_man         <- paste0(main_var, ":is_liberal:is_man")
  int_lib_notman      <- paste0(main_var, ":is_liberal:is_not_man")
  int_notlib_man      <- paste0(main_var, ":is_not_liberal:is_man")
  int_notlib_notman   <- paste0(main_var, ":is_not_liberal:is_not_man")
  
  # Model formulas
  f1 <- as.formula(paste0("mimesis ~ ", main_var, " + ", fe_part))
  f2 <- as.formula(paste0("mimesis ~ ", int_liberal, " + ", int_not_liberal, " + ", fe_part))
  f3 <- as.formula(paste0("mimesis ~ ", int_man, " + ", int_not_man, " + ", fe_part))
  f4 <- as.formula(paste0("mimesis ~ ",
                          int_lib_man, " + ", int_lib_notman, " + ",
                          int_notlib_man, " + ", int_notlib_notman, " + ",
                          fe_part))
  
  # Fit models ------------------------------------------------------------------
  m1 <- lm(f1, data = x)
  m2 <- lm(f2, data = x)
  m3 <- lm(f3, data = x)
  m4 <- lm(f4, data = x)
  models <- list("(1)" = m1, "(2)" = m2, "(3)" = m3, "(4)" = m4)
  
  # Cluster-robust VCOVs by user_id
  vcovs <- lapply(models, function(mod) sandwich::vcovCL(mod, cluster = x$user_id))
  names(vcovs) <- names(models)
  
  # Pretty label helper (strip "politics_" in output labels)
  pretty_label <- function(name) {
    nm <- sub("^aita_", "", name)            # drop politics_ prefix in table
    nm <- gsub(":", " x ", nm)
    nm <- gsub("_", "\\\\_", nm)
    nm
  }
  
  # Collect all coef names and FE rows
  all_coefs <- unique(unlist(lapply(models, function(m) names(coef(m)))))
  aita_rows <- sort(grep("^aita_", all_coefs, value = TRUE))
  
  # Coef ordering with dynamic main var (include aliases for interaction order)
  alias_pair <- function(a, b) c(paste(a, b, sep=":"), paste(b, a, sep=":"))
  alias_trip <- function(a, b, c) {
    unique(c(
      paste(a,b,c, sep=":"),
      paste(a,c,b, sep=":"),
      paste(b,a,c, sep=":"),
      paste(b,c,a, sep=":"),
      paste(c,a,b, sep=":"),
      paste(c,b,a, sep=":")
    ))
  }
  
  base_term <- main_var
  man_terms            <- alias_pair(main_var, "is_man")
  notman_terms         <- alias_pair(main_var, "is_not_man")
  lib_terms            <- alias_pair(main_var, "is_liberal")
  notlib_terms         <- alias_pair(main_var, "is_not_liberal")
  
  lib_man_terms        <- alias_trip(main_var, "is_liberal", "is_man")
  lib_notman_terms     <- alias_trip(main_var, "is_liberal", "is_not_man")
  notlib_man_terms     <- alias_trip(main_var, "is_not_liberal", "is_man")
  notlib_notman_terms  <- alias_trip(main_var, "is_not_liberal", "is_not_man")
  
  coef_order <- c(
    base_term,
    man_terms,
    notman_terms,
    lib_terms,
    notlib_terms,
    lib_man_terms,
    lib_notman_terms,
    notlib_man_terms,
    notlib_notman_terms,
    aita_rows,
    "(Intercept)"
  )
  
  # Build coef_map using pretty labels
  coef_map <- setNames(sapply(coef_order, pretty_label), coef_order)
  coef_map["(Intercept)"] <- "Intercept (No Context x Abortion)"
  
  # Stars, GOF, Adjusted R^2 ----------------------------------------------------
  star_levels <- c("***" = 0.01, "**" = 0.05, "*" = 0.10)
  adj_r2_vals <- sapply(models, function(m) round(summary(m)$adj.r.squared, 3))
  
  adj_r2_row <- tibble::tibble(
    term = "Adjusted R$^2$",
    `(1)` = adj_r2_vals[1],
    `(2)` = adj_r2_vals[2],
    `(3)` = adj_r2_vals[3],
    `(4)` = adj_r2_vals[4]
  )
  
  # Output path/tag -------------------------------------------------------------
  out_file <- paste0("../../long-context-eval/tables/mimesis_aita_", model_name, "_", table_tag, ".tex")
  
  # Build LaTeX table -----------------------------------------------------------
  tex <- modelsummary(
    models,
    vcov      = vcovs,
    coef_map  = coef_map,
    estimate  = "{estimate}{stars}",
    statistic = "({std.error})",
    stars     = star_levels,
    fmt       = 3,
    gof_map   = NULL,
    gof_omit  = ".*",
    escape    = FALSE,
    add_rows  = adj_r2_row,
    output    = "latex"
  )
  
  tex <- sub("\\\\begin\\{table\\}", paste0("\\\\begin{table}\n\\\\caption{Mimesis in Personal Advice: ", model_name, "}"), tex)
  
  # Midrules
  tex <- sub("(Intercept.*?\\\\\\\\)", "\\\\midrule\n\\1", tex, perl = TRUE)
  tex <- sub("(Adjusted R\\$\\^2\\$.*?\\\\\\\\)", "\\\\midrule\n\\1", tex, perl = TRUE)
  
  # Significance codes
  tex <- sub("(?=\\\\end\\{tabular\\})",
             "\n\\\\multicolumn{5}{r}{\\\\footnotesize\\\\textit{Significance codes:} *** \\$p<0.01\\$, ** \\$p<0.05\\$, * \\$p<0.10\\$} \\\\\\\\\n",
             tex, perl = TRUE)
  
  writeLines(tex, out_file)
  message("Wrote LaTeX table to: ", out_file)
}

# --- Examples -----------------------------------------------------------------
run_mimesis_models(main_var = "with_context", table_tag = "with_context", model_name = "gpt-4.1-mini-2025-04-14")
run_mimesis_models(main_var = "with_context", table_tag = "with_context", model_name = "claude-sonnet-4-20250514")
run_mimesis_models(main_var = "understanding", table_tag = "understanding", model_name = "gpt-4.1-mini-2025-04-14")
run_mimesis_models(main_var = "understanding", table_tag = "understanding", model_name = "claude-sonnet-4-20250514")

