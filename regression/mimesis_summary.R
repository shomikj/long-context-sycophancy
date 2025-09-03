# --- Libraries ---------------------------------------------------------------
library(sandwich)
library(lmtest)
library(modelsummary)
library(tibble)
library(dplyr)
library(stringr)

# --- Helpers -----------------------------------------------------------------
`%||%` <- function(a, b) if (is.null(a)) b else a

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
first_present <- function(candidates, universe) {
  present <- intersect(candidates, universe)
  if (length(present) == 0) character(0) else present[1]
}

latex_escape_underscores <- function(x) gsub("_", "\\\\_", x, fixed = TRUE)

star_code <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.10) return("*")
  ""
}

fmt_num <- function(x, digits = 3) {
  if (is.na(x)) return(NA_character_)
  sprintf(paste0("%.", digits, "f"), x)
}

# Two-line entry: first row = estimate{stars}; second row = (SE)
fmt_entry_pair <- function(est, se, p, digits = 3) {
  if (is.na(est) || is.na(se)) return(c("—", ""))  # keeps row structure even if missing
  est_str <- paste0("\\num{", fmt_num(est, digits), "}", star_code(p))
  se_str  <- paste0("(\\num{", fmt_num(se, digits), "})")
  c(est_str, se_str)
}

# --- Core: fit the four specifications for one (model_name, task_name) -------
fit_specs <- function(dat, y_var, x_var) {
  f1 <- as.formula(paste(y_var, "~", x_var, "+", "factor(prompt_id)"))
  f2 <- as.formula(paste(
    y_var, "~",
    paste(x_var, c(":is_man", ":is_woman"), sep = "", collapse = " + "),
    "+ factor(prompt_id)"
  ))
  f3 <- as.formula(paste(
    y_var, "~",
    paste(x_var, c(":has_left_politics", ":has_right_politics"), sep = "", collapse = " + "),
    "+ factor(prompt_id)"
  ))
  f4 <- as.formula(paste(
    y_var, "~",
    paste(
      x_var,
      c(":is_man:has_left_politics",
        ":is_man:has_right_politics",
        ":is_woman:has_left_politics",
        ":is_woman:has_right_politics"),
      sep = "", collapse = " + "),
    "+ factor(prompt_id)"
  ))
  
  m1 <- lm(f1, data = dat)
  m2 <- lm(f2, data = dat)
  m3 <- lm(f3, data = dat)
  m4 <- lm(f4, data = dat)
  
  models <- list(m1 = m1, m2 = m2, m3 = m3, m4 = m4)
  vcovs  <- lapply(models, function(mod) sandwich::vcovCL(mod, cluster = ~ user_id))
  list(models = models, vcovs = vcovs)
}

# Extract estimate+SE (cluster-robust) for the right spec, returned as two lines
extract_pair <- function(models, vcovs, spec, term_aliases, digits = 3) {
  mod <- models[[spec]]
  V   <- vcovs[[spec]]
  if (is.null(mod) || is.null(V)) return(c("—", ""))
  
  ct <- lmtest::coeftest(mod, vcov. = V)
  rown <- rownames(ct)
  term <- first_present(term_aliases, rown)
  if (length(term) == 0) return(c("—", ""))
  
  est  <- unname(ct[term, "Estimate"])
  se   <- unname(ct[term, "Std. Error"])
  pcol <- which(colnames(ct) %in% c("Pr(>|t|)", "Pr(>|z|)", "p"))
  pval <- if (length(pcol)) unname(ct[term, pcol[1]]) else {
    tstat <- est / se
    2 * (1 - pnorm(abs(tstat)))
  }
  fmt_entry_pair(est, se, pval, digits = digits)
}

# For one (model_name, task_name), build a 2-row-per-term column (length = 18)
build_column <- function(dat, y_var, x_var, digits = 3) {
  fit <- fit_specs(dat, y_var, x_var)
  
  # Alias sets
  t_x             <- x_var
  t_x_man         <- alias_pair(x_var, "is_man")
  t_x_woman       <- alias_pair(x_var, "is_woman")
  t_x_left        <- alias_pair(x_var, "has_left_politics")
  t_x_right       <- alias_pair(x_var, "has_right_politics")
  t_x_man_left    <- alias_trip(x_var, "is_man", "has_left_politics")
  t_x_man_right   <- alias_trip(x_var, "is_man", "has_right_politics")
  t_x_woman_left  <- alias_trip(x_var, "is_woman", "has_left_politics")
  t_x_woman_right <- alias_trip(x_var, "is_woman", "has_right_politics")
  
  row_defs <- list(
    list(key = x_var,                                 aliases = t_x,             spec = "m1"),
    list(key = paste(x_var, "is_man",            sep=":"), aliases = t_x_man,         spec = "m2"),
    list(key = paste(x_var, "is_woman",          sep=":"), aliases = t_x_woman,       spec = "m2"),
    list(key = paste(x_var, "has_left_politics", sep=":"), aliases = t_x_left,        spec = "m3"),
    list(key = paste(x_var, "has_right_politics",sep=":"), aliases = t_x_right,       spec = "m3"),
    list(key = paste(x_var, "is_man",   "has_left_politics",  sep=":"), aliases = t_x_man_left,    spec = "m4"),
    list(key = paste(x_var, "is_man",   "has_right_politics", sep=":"), aliases = t_x_man_right,   spec = "m4"),
    list(key = paste(x_var, "is_woman", "has_left_politics",  sep=":"), aliases = t_x_woman_left,  spec = "m4"),
    list(key = paste(x_var, "is_woman", "has_right_politics", sep=":"), aliases = t_x_woman_right, spec = "m4")
  )
  
  # Build column values: for each term, append (estimate_line, se_line)
  entries <- c()
  for (rd in row_defs) {
    pair <- extract_pair(fit$models, fit$vcovs, rd$spec, rd$aliases, digits = digits)
    entries <- c(entries, pair)
  }
  entries
}

# --- Main: Build the single 4-column table -----------------------------------
build_combined_table <- function(cfg, digits = 3) {
  # Defaults & required
  stopifnot(is.list(cfg))
  cfg$data_dir <- cfg$data_dir %||% "."
  cfg$out_dir  <- cfg$out_dir  %||% file.path(cfg$data_dir, "..", "..", "long-context-eval", "tables")
  required <- c("file_name", "y_var", "x_var")
  missing  <- setdiff(required, names(cfg))
  if (length(missing)) stop("Missing required cfg fields: ", paste(missing, collapse = ", "))
  
  # Paths
  csv_path <- file.path(cfg$data_dir, cfg$file_name)
  out_file <- cfg$out_file %||% file.path(
    cfg$out_dir,
    paste0(cfg$y_var, "_", cfg$x_var, ".tex")
  )
  
  # Load once
  dat <- read.csv(csv_path, stringsAsFactors = FALSE, header = TRUE)
  
  # ---------- Pretty labels (your reference, with underscore escaping) ----------
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
  
  # Define the four columns in the requested order
  COLS <- list(
    "Claude x politics" = list(model_name = "claude-sonnet-4-20250514", task_name = "politics"),
    "GPT x politics"    = list(model_name = "gpt-4.1-mini-2025-04-14", task_name = "politics"),
    "Claude x aita"     = list(model_name = "claude-sonnet-4-20250514", task_name = "aita"),
    "GPT x aita"        = list(model_name = "gpt-4.1-mini-2025-04-14", task_name = "aita")
  )
  
  # Build each column by subsetting data, fitting specs, and extracting values
  columns_list <- lapply(names(COLS), function(colname) {
    spec <- COLS[[colname]]
    dat_sub <- dplyr::filter(dat, model == spec$model_name, task == spec$task_name)
    build_column(dat_sub, cfg$y_var, cfg$x_var, digits = digits)
  })
  names(columns_list) <- names(COLS)
  
  # ----- Build the Term column: 2 rows per term with escaped labels ----------
  base_terms <- c(
    cfg$x_var,
    paste(cfg$x_var, "is_man",            sep=":"),
    paste(cfg$x_var, "is_woman",          sep=":"),
    paste(cfg$x_var, "has_left_politics", sep=":"),
    paste(cfg$x_var, "has_right_politics",sep=":"),
    paste(cfg$x_var, "is_man",   "has_left_politics",  sep=":"),
    paste(cfg$x_var, "is_man",   "has_right_politics", sep=":"),
    paste(cfg$x_var, "is_woman", "has_left_politics",  sep=":"),
    paste(cfg$x_var, "is_woman", "has_right_politics", sep=":")
  )
  
  # Use "\\," as the SE-row label so the row is preserved in LaTeX but appears blank
  term_rows <- as.vector(rbind(
    vapply(base_terms, pretty_label, character(1)),  # label row (escaped)
    rep("\\,", length(base_terms))                   # SE row (kept, effectively blank)
  ))
  
  # ---- Stack into a data frame with proper (unique) column names ----
  cols_named <- setNames(
    lapply(names(columns_list), function(nm) unname(columns_list[[nm]])),
    names(columns_list)
  )
  
  df <- tibble::tibble(
    Term = term_rows,
    !!!cols_named
  )
  
  # Build LaTeX title with underscores escaped
  title_tex <- paste0(
    latex_escape_underscores(cfg$y_var),
    " ~ ",
    latex_escape_underscores(cfg$x_var),
    " \\textbar{} combined specs \\& configurations"
  )
  
  # Output LaTeX
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  tex <- modelsummary::datasummary_df(
    df,
    title  = title_tex,
    output = "latex",
    escape = FALSE,  # we inject \num and LaTeX helpers; labels already escaped
    notes  = list("\\textit{Significance:} *** $p<0.01$, ** $p<0.05$, * $p<0.10$")
  )
  
  writeLines(tex, out_file)
  message("Wrote LaTeX table to: ", out_file)
  
  invisible(list(tex = tex, table = df, out_file = out_file))
}

# ---------------- Run it ------------------------------------------------------
base_cfg <- list(
  data_dir   = "/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression",
  file_name  = "mimesis.csv",
  y_var      = "mimesis",
  x_var      = "understanding",
  out_dir    = "/Users/shomik/Documents/MIT/Projects/Alignment_Drift/long-context-eval/tables"
)

# Build the single 4-column table
res_combined <- build_combined_table(base_cfg, digits = 3)

