# Needed packages
library(sandwich)  # vcovCL
library(lmtest)    # coeftest

# Helper: p-value -> stars
p_to_stars <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  ""
}

# ---- Function ----
# Runs the interaction regressions for all X[0-9]-prefixed topic columns
# and returns a tidy data frame for the given model_name & task_name.
topic_interactions <- function(file_name, model_name, task_name) {
  # read + filter
  x <- read.csv(file = file_name, stringsAsFactors = FALSE, header = TRUE)
  x <- subset(x, model == model_name & task == task_name)
  
  # catch empty subset early
  if (nrow(x) == 0) {
    return(data.frame(
      model_name = character(0),
      task_name  = character(0),
      topic_column = character(0),
      coefficient  = numeric(0),
      standard_error = numeric(0),
      p_value     = numeric(0),
      statistical_significance = character(0),
      stringsAsFactors = FALSE
    ))
  }
  
  # all columns that begin with X and a digit
  topic_cols <- grep("^X\\d", names(x), value = TRUE)
  
  res_list <- lapply(topic_cols, function(col) {
    # mimesis ~ with_context:<topic> + factor(prompt_id)
    fml <- as.formula(paste0("mimesis ~ with_context:", col, " + factor(prompt_id)"))
    
    m  <- lm(fml, data = x)
    VC <- vcovCL(m, cluster = ~ user_id)
    ct <- coeftest(m, vcov. = VC)
    
    # pick the interaction row (order can swap)
    rn      <- rownames(ct)
    targets <- which(rn %in% c(paste0("with_context:", col), paste0(col, ":with_context")))
    
    if (length(targets) == 0) {
      return(data.frame(
        topic_column             = col,
        coefficient              = NA_real_,
        standard_error           = NA_real_,
        p_value                  = NA_real_,
        statistical_significance = NA_character_,
        stringsAsFactors = FALSE
      ))
    }
    
    i   <- targets[1]
    est <- unname(ct[i, 1])
    se  <- unname(ct[i, 2])
    pv  <- unname(ct[i, 4])
    
    data.frame(
      topic_column             = col,
      coefficient              = est,
      standard_error           = se,
      p_value                  = pv,
      statistical_significance = p_to_stars(pv),
      stringsAsFactors = FALSE
    )
  })
  
  df <- do.call(rbind, res_list)
  
  # add identifiers and order columns
  df$model_name <- model_name
  df$task_name  <- task_name
  df <- df[, c("model_name","task_name","topic_column",
               "coefficient","standard_error","p_value","statistical_significance")]
  
  df
}

# ---- Example: call the function 4 times and aggregate ----
# Replace the model/task pairs below with the combinations you want to run.
file_name <- "mimesis_topics.csv"

runs <- list(
  list(model = "claude-sonnet-4-20250514", task = "aita"),
  list(model = "claude-sonnet-4-20250514", task = "politics"),
  list(model = "gpt-4.1-mini-2025-04-14",     task = "aita"),
  list(model = "gpt-4.1-mini-2025-04-14",     task = "politics")
)

results_all <- do.call(
  rbind,
  lapply(runs, function(p) topic_interactions(file_name, p$model, p$task))
)

# View the aggregated results
write.csv(results_all, file = "topic_coefficients.csv", row.names = FALSE)

