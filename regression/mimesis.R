# --- Libraries ---------------------------------------------------------------
library(sandwich)
library(lmtest)
library(modelsummary)
library(tibble)

# --- Function ----------------------------------------------------------------
#' Build clustered-robust regression table and write LaTeX
#'
#' @param cfg named list with fields:
#'   - data_dir   : character, folder containing the CSV (default = ".")
#'   - file_name  : character, CSV filename (required)
#'   - model_name : character, filter for `model` column (required)
#'   - task_name  : character, filter for `task` column (required)
#'   - y_var      : character, dependent variable (required)
#'   - x_var      : character, key regressor prefix used in interactions (required)
#'   - out_dir    : character, where to write the .tex file (default builds the old path)
#'   - out_file   : character, explicit output path; if missing, composed from pieces
#'
#' @return a list with:
#'   - tex     : the LaTeX table string
#'   - models  : list of lm objects
#'   - vcovs   : list of clustered-robust vcov matrices
#'   - gof     : tibble of custom GOF rows
#'   - out_file: path written to (invisibly returned)
#'
# --- Function ----------------------------------------------------------------
build_regression_table <- function(cfg) {
  # Define helper operators up front
  `%||%` <- function(a, b) if (is.null(a)) b else a
  `%+%`  <- function(a, b) paste0(a, b)
  
  # ---------- Defaults & path handling ----------
  stopifnot(is.list(cfg))
  cfg$data_dir   <- cfg$data_dir   %||% "."
  cfg$out_dir    <- cfg$out_dir    %||% file.path(cfg$data_dir, "..", "..", "long-context-eval", "tables")
  
  required <- c("file_name", "model_name", "task_name", "y_var", "x_var")
  missing  <- setdiff(required, names(cfg))
  if (length(missing)) stop("Missing required cfg fields: ", paste(missing, collapse = ", "))
  
  if (is.null(cfg$out_file)) {
    cfg$out_file <- file.path(
      cfg$out_dir,
      paste0(cfg$y_var, "_", cfg$x_var, "_", cfg$task_name, "_", cfg$model_name, ".tex")
    )
  }
  
  # ---------- Data ----------
  csv_path <- file.path(cfg$data_dir, cfg$file_name)
  x <- read.csv(file = csv_path, stringsAsFactors = FALSE, header = TRUE)
  x <- subset(x, model == cfg$model_name)
  x <- subset(x, task  == cfg$task_name)
  
  # ---------- Models (unchanged) ----------
  f1 <- as.formula(paste(cfg$y_var, "~", cfg$x_var, "+", "factor(prompt_id)"))
  f2 <- as.formula(paste(
    cfg$y_var, "~",
    paste(cfg$x_var, c(":is_man", ":is_woman"), sep = "", collapse = " + "),
    "+ factor(prompt_id)"
  ))
  f3 <- as.formula(paste(
    cfg$y_var, "~",
    paste(cfg$x_var, c(":has_left_politics", ":has_right_politics"), sep = "", collapse = " + "),
    "+ factor(prompt_id)"
  ))
  f4 <- as.formula(paste(
    cfg$y_var, "~",
    paste(
      cfg$x_var,
      c(":is_man:has_left_politics",
        ":is_man:has_right_politics",
        ":is_woman:has_left_politics",
        ":is_woman:has_right_politics"),
      sep = "", collapse = " + "),
    "+ factor(prompt_id)"
  ))
  
  m1 <- lm(f1, data = x)
  m2 <- lm(f2, data = x)
  m3 <- lm(f3, data = x)
  m4 <- lm(f4, data = x)
  
  models <- list("(1)" = m1, "(2)" = m2, "(3)" = m3, "(4)" = m4)
  
  # ---------- Cluster-robust vcovs (by user_id) ----------
  vcovs <- lapply(models, function(mod) sandwich::vcovCL(mod, cluster = ~ user_id))
  
  # ---------- Row ordering logic ----------
  alias_pair <- function(a, b) unique(c(paste(a,b,sep=":"), paste(b,a,sep=":")))
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
  
  all_coefs <- unique(unlist(lapply(models, function(m) names(coef(m)))))
  fe_terms       <- sort(grep("^factor\\(prompt_id\\)", all_coefs, value = TRUE))
  intercept_term <- "(Intercept)"
  
  t_x             <- cfg$x_var
  t_x_man         <- alias_pair(cfg$x_var, "is_man")
  t_x_woman       <- alias_pair(cfg$x_var, "is_woman")
  t_x_left        <- alias_pair(cfg$x_var, "has_left_politics")
  t_x_right       <- alias_pair(cfg$x_var, "has_right_politics")
  t_x_man_left    <- alias_trip(cfg$x_var, "is_man", "has_left_politics")
  t_x_man_right   <- alias_trip(cfg$x_var, "is_man", "has_right_politics")
  t_x_woman_left  <- alias_trip(cfg$x_var, "is_woman", "has_left_politics")
  t_x_woman_right <- alias_trip(cfg$x_var, "is_woman", "has_right_politics")
  
  first_present <- function(candidates, universe) {
    present <- intersect(candidates, universe)
    if (length(present) == 0) character(0) else present
  }
  
  ordered_primary <- c(
    first_present(t_x,             all_coefs),
    first_present(t_x_man,         all_coefs),
    first_present(t_x_woman,       all_coefs),
    first_present(t_x_left,        all_coefs),
    first_present(t_x_right,       all_coefs),
    first_present(t_x_man_left,    all_coefs),
    first_present(t_x_man_right,   all_coefs),
    first_present(t_x_woman_left,  all_coefs),
    first_present(t_x_woman_right, all_coefs)
  )
  
  coef_order <- c(ordered_primary, fe_terms, intercept_term)
  
  # ---------- Pretty labels ----------
  pretty_label <- function(name) {
    if (grepl("^factor\\(prompt_id\\)", name)) {
      nm <- sub("^factor\\(prompt_id\\)", "", name) # raw FE label only
      nm <- gsub("_", "\\\\_", nm)
      return(nm)
    }
    parts <- strsplit(name, ":", fixed = TRUE)[[1]]
    if (length(parts) > 1) {
      priority <- c(cfg$x_var, "is_man", "is_woman", "has_left_politics", "has_right_politics")
      idx <- match(parts, priority)
      ord <- order(is.na(idx), idx, seq_along(parts))
      nm <- paste(parts[ord], collapse = " x ")
    } else {
      nm <- name
    }
    gsub("_", "\\\\_", nm)
  }
  coef_map <- setNames(vapply(coef_order, pretty_label, character(1)), coef_order)
  coef_map[intercept_term] <- "Intercept"
  
  # ---------- Custom GOF rows ----------
  adj_r2_vals <- sapply(models, function(m) summary(m)$adj.r.squared)
  nobs_vals   <- sapply(models, nobs)
  
  uniq_users_per_model <- function(m) {
    used <- rep(TRUE, nrow(x))
    if (!is.null(m$na.action)) used[as.integer(m$na.action)] <- FALSE
    length(unique(x$user_id[used]))
  }
  uniq_users_vals <- sapply(models, uniq_users_per_model)
  
  nobs_str  <- as.character(as.integer(nobs_vals))
  users_str <- as.character(as.integer(uniq_users_vals))
  r2_str    <- sprintf("%.3f", adj_r2_vals)
  
  gof_rows <- tibble(
    term = c("N (Observations)", "N (Participants)", "Adjusted R$^2$"),
    `(1)` = c(nobs_str[1],  users_str[1],  r2_str[1]),
    `(2)` = c(nobs_str[2],  users_str[2],  r2_str[2]),
    `(3)` = c(nobs_str[3],  users_str[3],  r2_str[3]),
    `(4)` = c(nobs_str[4],  users_str[4],  r2_str[4])
  )
  
  # ---------- Build LaTeX table ----------
  star_levels <- c("***" = 0.01, "**" = 0.05, "*" = 0.10)
  
  tex <- modelsummary(
    models,
    vcov       = vcovs,
    coef_map   = coef_map,
    coef_order = coef_order,
    estimate   = "{estimate}{stars}",
    statistic  = "({std.error})",
    stars      = star_levels,
    fmt        = 3,
    gof_map    = NULL,
    gof_omit   = ".*",
    escape     = FALSE,
    add_rows   = gof_rows,
    output     = "latex"
  )
  
  # Caption
  tex <- sub("\\\\begin\\{table\\}",
             paste0("\\\\begin{table}\n",
                    "\\\\caption{", cfg$y_var, " x ", cfg$x_var,
                    " (task: ", cfg$task_name, ", model: ", cfg$model_name, ")}"),
             tex)
  
  # ---- Midrules ----
  escape_regex <- function(s) {
    gsub("([\\.^$|()\\[\\]{}*+?\\\\])", "\\\\\\1", s, perl = TRUE)
  }
  
  if (length(fe_terms) > 0) {
    fe_label_first <- unname(coef_map[fe_terms[1]])
    fe_pat <- paste0("(", escape_regex(fe_label_first), ".*?\\\\\\\\)")
    tex <- sub(fe_pat, "\\\\midrule\n\\1", tex, perl = TRUE)
  }
  
  intercept_block_pat <- "(?s)(Intercept.*?\\\\\\\\\\s*\\n\\s*&\\s*\\(.*?\\)\\s*\\\\\\\\)"
  tex <- sub(intercept_block_pat, "\\\\midrule\n\\1\n\\\\midrule\n", tex, perl = TRUE)
  
  tex <- sub("(?=\\\\end\\{tabular\\})",
             "\n\\\\multicolumn{5}{r}{\\\\footnotesize\\\\textit{Significance codes:} *** $p<0.01$, ** $p<0.05$, * $p<0.10$} \\\\\\\\\n",
             tex, perl = TRUE)
  
  # ---------- Write file & return ----------
  dir.create(dirname(cfg$out_file), showWarnings = FALSE, recursive = TRUE)
  writeLines(tex, cfg$out_file)
  message("Wrote LaTeX table to: ", cfg$out_file)
  
  invisible(list(
    tex      = tex,
    models   = models,
    vcovs    = vcovs,
    gof      = gof_rows,
    out_file = cfg$out_file
  ))
}

# Assume the function build_regression_table() from earlier is already loaded

# Shared config parts
base_cfg <- list(
  data_dir   = "/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression",
  file_name  = "sycophancy.csv",
  y_var      = "sycophancy",
  x_var      = "with_context",
  out_dir    = "/Users/shomik/Documents/MIT/Projects/Alignment_Drift/long-context-eval/tables"
)

# 1) gpt-4.1-mini-2025-04-14, aita
res1 <- build_regression_table(modifyList(base_cfg, list(
  model_name = "gpt-4.1-mini-2025-04-14",
  task_name  = "aita"
)))

# 2) gpt-4.1-mini-2025-04-14, politics
#res2 <- build_regression_table(modifyList(base_cfg, list(
#  model_name = "gpt-4.1-mini-2025-04-14",
#  task_name  = "politics"
#)))

# 3) claude-sonnet-4-20250514, aita
res3 <- build_regression_table(modifyList(base_cfg, list(
  model_name = "claude-sonnet-4-20250514",
  task_name  = "aita"
)))

# 4) claude-sonnet-4-20250514, politics
#res4 <- build_regression_table(modifyList(base_cfg, list(
#  model_name = "claude-sonnet-4-20250514",
#  task_name  = "politics"
#)))
