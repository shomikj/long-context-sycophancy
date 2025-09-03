library(foreign)
library(estimatr)
library(sandwich)
library(lmtest)
library(stargazer)
library(xtable)
library(arm)
library(grDevices)
library("dplyr")
library("car")
library(fixest)
library(modelsummary)

setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

run_models_for <- function(main_var = "with_context", model_name="claude-sonnet-4-20250514") {
  x <- read.csv(file = "aita_sycophancy.csv", stringsAsFactors = F, header = T)
  x <- subset(x, model==model_name)
  if (main_var == "understanding") {
    x <- subset(x, passed_attention == 1)
  }
  # ---- model fits (dynamic main_var) ----
  controls <- c(
    "aita_1ll3bu4","aita_1l3z8cc","aita_1jwnsuz","aita_1f52bvv",
    "aita_1flfh0r","aita_1l0wnzy","aita_1hi8obt","aita_1ilxjuy","aita_1ej4a0t"
  )
  controls_str <- paste(controls, collapse = " + ")
  
  f1 <- as.formula(paste0("sycophancy ~ ", main_var, " + ", controls_str))
  f2 <- as.formula(paste0(
    "sycophancy ~ ", main_var, ":is_man + ", main_var, ":is_not_man + ", controls_str
  ))
  f3 <- as.formula(paste0(
    "sycophancy ~ ", main_var, ":is_liberal + ", main_var, ":is_not_liberal + ", controls_str
  ))
  f4 <- as.formula(paste0(
    "sycophancy ~ ",
    main_var, ":is_man:is_liberal + ",
    main_var, ":is_not_man:is_liberal + ",
    main_var, ":is_not_liberal:is_man + ",
    main_var, ":is_not_liberal:is_not_man + ",
    controls_str
  ))
  
  m1 <- lm(f1, data = x)
  m2 <- lm(f2, data = x)
  m3 <- lm(f3, data = x)
  m4 <- lm(f4, data = x)
  
  models <- list("(1)" = m1, "(2)" = m2, "(3)" = m3, "(4)" = m4)
  vcovs <- lapply(models, function(mod) sandwich::vcovCL(mod, cluster = x$user_id))
  names(vcovs) <- names(models)
  
  # ---- rows that involve the main var (dynamic) ----
  main_rows <- c(
    main_var,
    paste0(main_var, ":is_man"),
    paste0(main_var, ":is_not_man"),
    paste0(main_var, ":is_liberal"),
    paste0(main_var, ":is_not_liberal"),
    paste0(main_var, ":is_man:is_liberal"),
    paste0(main_var, ":is_not_man:is_liberal"),
    paste0(main_var, ":is_man:is_not_liberal"),
    paste0(main_var, ":is_not_man:is_not_liberal")
  )
  
  all_coefs <- unique(unlist(lapply(models, function(m) names(coef(m)))))
  aita_rows <- sort(grep("^aita_", all_coefs, value = TRUE))
  
  # Duplicate entries to handle alphabetic ordering of interaction terms (dynamic)
  coef_order <- c(
    main_var,
    paste0(main_var, ":is_man"),
    paste0(main_var, ":is_not_man"),
    paste0(main_var, ":is_liberal"),
    paste0(main_var, ":is_not_liberal"),
    paste0(main_var, ":is_man:is_liberal"),
    paste0(main_var, ":is_liberal:is_man"),       # alias for m4 naming
    paste0(main_var, ":is_not_man:is_liberal"),
    paste0(main_var, ":is_liberal:is_not_man"),   # alias for m4 naming
    paste0(main_var, ":is_man:is_not_liberal"),
    paste0(main_var, ":is_not_man:is_not_liberal"),
    aita_rows,
    "(Intercept)"
  )
  
  pretty_label <- function(name) {
    # Replace ":" with " x ", then escape underscores for LaTeX
    gsub("_", "\\\\_", gsub(":", " x ", name))
  }
  
  coef_map <- setNames(sapply(coef_order, pretty_label), coef_order)
  
  # Give the aliases the same pretty label (dynamic)
  coef_map[paste0(main_var, ":is_liberal:is_man")]     <- coef_map[paste0(main_var, ":is_man:is_liberal")]
  coef_map[paste0(main_var, ":is_liberal:is_not_man")] <- coef_map[paste0(main_var, ":is_not_man:is_liberal")]
  coef_map["(Intercept)"] <- "Intercept (zero\\_shot x aita\\_1gyx5dc)"
  
  pretty_label <- function(name) {
    # Replace ":" with " x ", then escape underscores for LaTeX
    gsub("_", "\\\\_", gsub(":", " x ", name))
  }
  
  coef_map <- setNames(sapply(coef_order, pretty_label), coef_order)
  coef_map["(Intercept)"] <- "Intercept (zero\\_shot x aita\\_1gyx5dc)"
  
  star_levels <- c("***" = 0.01, "**" = 0.05, "*" = 0.10)
  
  gof_map <- data.frame(
    raw   = "adj.r.squared",
    clean = "Adjusted R^2",
    fmt   = 3,
    stringsAsFactors = FALSE
  )
  
  # Extract Adjusted R² for each model
  adj_r2_vals <- sapply(models, function(m) round(summary(m)$adj.r.squared, 3))
  
  # Create a tibble with the custom row
  adj_r2_row <- tibble::tibble(
    term = "Adjusted R$^2$",
    `(1)` = adj_r2_vals[1],
    `(2)` = adj_r2_vals[2],
    `(3)` = adj_r2_vals[3],
    `(4)` = adj_r2_vals[4]
  )
  
  # Extract Adjusted R² for each model (duplicated intentionally to keep original logic)
  adj_r2_vals <- sapply(models, function(m) round(summary(m)$adj.r.squared, 3))
  
  # Create a tibble with the custom row (duplicated intentionally to keep original logic)
  adj_r2_row <- tibble::tibble(
    term = "Adjusted R$^2$",
    `(1)` = adj_r2_vals[1],
    `(2)` = adj_r2_vals[2],
    `(3)` = adj_r2_vals[3],
    `(4)` = adj_r2_vals[4]
  )
  
  # Now build the table
  tex <- modelsummary(
    models,
    vcov       = vcovs,
    coef_map   = coef_map,
    estimate   = "{estimate}{stars}",
    statistic  = "({std.error})",
    stars      = star_levels,
    fmt        = 3,
    gof_map    = NULL,   # don't let gof_map auto-add
    gof_omit   = ".*",   # hide default GOF
    escape     = FALSE,
    add_rows   = adj_r2_row,
    output     = "latex",
    file       = "regression_table.tex"
  )
  
  tex <- sub(
    "(Intercept.*?\\\\\\\\)",
    "\\\\midrule\n\\1",
    tex,
    perl = TRUE
  )
  
  # Add a midrule before the Adjusted R^2 row
  tex <- sub(
    "(Adjusted R\\$\\^2\\$.*?\\\\\\\\)",
    "\\\\midrule\n\\1",
    tex,
    perl = TRUE
  )
  
  # Insert significance codes before \end{tabular}
  tex <- sub(
    "(?=\\\\end\\{tabular\\})",
    "\n\\\\multicolumn{5}{r}{\\\\footnotesize\\\\textit{Significance codes:} *** \\$p<0.01\\$, ** \\$p<0.05\\$, * \\$p<0.10\\$} \\\\\n",
    tex,
    perl = TRUE
  )
  
  caption_text <- paste0("Sycophancy in Personal Advice: ", model_name)
  tex <- sub(
    "(\\\\begin\\{table\\}(?:\\[[^]]*\\])?\\s*)",
    paste0("\\1\\\\caption{", caption_text, "}"),
    tex,
    perl = TRUE
  )
  
  out_file <- paste0("../../long-context-eval/tables/sycophancy_", model_name, "_", main_var, ".tex")
  writeLines(tex, out_file)
  
  invisible(NULL)
}

# ---- run with the default main variable ----
run_models_for("with_context", "gpt-4.1-mini-2025-04-14")
run_models_for("with_context", "claude-sonnet-4-20250514")
run_models_for("understanding", "gpt-4.1-mini-2025-04-14")
run_models_for("understanding", "claude-sonnet-4-20250514")

