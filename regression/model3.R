library(sandwich)
library(lmtest)
library(stargazer)

# --- Parameters ---
setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

file_name <- "mimesis.csv"
model_name <- "claude-sonnet-4-20250514"
#model_name <- "gpt-4.1-mini-2025-04-14"
task_name <- "politics"
y_var <- "mimesis"


# --- Load & filter data ---
x <- read.csv(file = file_name, stringsAsFactors = FALSE, header = TRUE)
x <- subset(x, model == model_name)
x <- subset(x, task  == task_name)

# --- Fit model with prompt fixed effects ---
m1 <- lm(as.formula(paste(y_var, "~", "context + context:understanding + is_man + has_left_politics + context:is_man + context:has_left_politics + ", "factor(prompt_id)")), data = x)
#m1 <- lm(as.formula(paste(y_var, "~", "context + context:understanding  + context:is_man + context:has_left_politics + ", "factor(prompt_id)")), data = x)

# Cluster-robust VCOV by user_id
m1_V_cl <- vcovCL(m1, cluster = ~ user_id)

cl_se <- sqrt(diag(m1_V_cl))

# Print table with stars
#stargazer(m1,
#          type = "text",
#          se = list(cl_se),
#          title = "Regression Results with Clustered SEs",
#          dep.var.labels = y_var,
#          star.cutoffs = c(0.1, 0.05, 0.01),
#          notes = "Clustered standard errors by user_id")


library(sandwich)
library(lmtest)
library(stargazer)

# --- Parameters ---
setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

file_name  <- "mimesis.csv"
#model_name <- "claude-sonnet-4-20250514"
model_name <- "gpt-4.1-mini-2025-04-14"
task_name  <- "politics"
y_var      <- "mimesis"

# --- Load & filter data ---
x <- read.csv(file = file_name, stringsAsFactors = FALSE, header = TRUE)
x <- subset(x, model == model_name)
x <- subset(x, task  == task_name)

# Make sure types are right for modeling/prediction
# (adjust these as needed based on your data)
x$prompt_id         <- factor(x$prompt_id)
x$context           <- as.numeric(x$context)
x$understanding     <- as.numeric(x$understanding)
x$is_man            <- as.numeric(x$is_man)
x$has_left_politics <- as.numeric(x$has_left_politics)

# --- Fit model with prompt fixed effects ---
fml <- as.formula(paste(
  y_var, "~",
  "context + context:understanding + is_man + has_left_politics +",
  "context:is_man + context:has_left_politics + prompt_id"
))
m1 <- lm(fml, data = x)

# Cluster-robust VCOV by user_id
m1_V_cl <- vcovCL(m1, cluster = ~ user_id)
cl_se   <- sqrt(diag(m1_V_cl))

# Print table with stars
stargazer(
  m1,
  type = "text",
  se = list(cl_se),
  title = "Regression Results with Clustered SEs",
  dep.var.labels = y_var,
  star.cutoffs = c(0.1, 0.05, 0.01),
  notes = "Clustered standard errors by user_id"
)

# --------------------------------------------------------------------
# Prediction utilities (plug in values and evaluate the model)
# --------------------------------------------------------------------

# Build a safe model matrix for newdata using m1's terms/contrasts/xlevels
.mm_from_model <- function(model, newdata) {
  TT   <- delete.response(terms(model))
  xlev <- model$xlevels
  # ensure categorical levels are known
  for (nm in names(xlev)) {
    if (nm %in% names(newdata)) {
      newdata[[nm]] <- factor(newdata[[nm]], levels = xlev[[nm]])
    }
  }
  model.matrix(TT, newdata, contrasts.arg = model$contrasts, xlev = xlev)
}

# Predict with custom (cluster-robust) vcov; returns fit, SE, CI
predict_cr <- function(model, vcov_mat, newdata, level = 0.95) {
  X    <- .mm_from_model(model, newdata)
  beta <- coef(model)
  fit  <- as.numeric(X %*% beta)
  V    <- X %*% vcov_mat %*% t(X)
  se   <- sqrt(pmax(diag(V), 0))
  z    <- qnorm(1 - (1 - level) / 2)
  lwr  <- fit - z * se
  upr  <- fit + z * se
  cbind(newdata, .fitted = fit, .se = se, .lwr = lwr, .upr = upr)
}

# A helper that fills in any missing columns with sensible defaults
# and validates prompt_id against the model's levels.
newdata_template <- function(
    context           = 0,
    understanding     = 0,
    is_man            = 0,
    has_left_politics = 0,
    prompt_id         = NULL
) {
  if (is.null(prompt_id)) {
    # default to the most frequent prompt_id in the estimation sample
    prompt_id <- names(sort(table(x$prompt_id), decreasing = TRUE))[1]
  }
  df <- data.frame(
    context = context,
    understanding = understanding,
    is_man = is_man,
    has_left_politics = has_left_politics,
    prompt_id = prompt_id,
    stringsAsFactors = FALSE
  )
  # coerce to factor with the model's levels
  df$prompt_id <- factor(df$prompt_id, levels = m1$xlevels$prompt_id)
  df
}

# --------------------------------------------------------------------
# Examples you can run (edit the values to "plug in" what you want)
# --------------------------------------------------------------------


# 2) Compare two scenarios head-to-head (e.g., left vs not-left)
ex2 <- rbind(
  newdata_template(context = 0, understanding = 0, is_man = 1, has_left_politics = 1, prompt_id = levels(x$prompt_id)[1]),
  newdata_template(context = 0, understanding = 0, is_man = 1, has_left_politics = 0, prompt_id = levels(x$prompt_id)[1]),
  newdata_template(context = 0, understanding = 0, is_man = 0, has_left_politics = 1, prompt_id = levels(x$prompt_id)[1]),
  newdata_template(context = 0, understanding = 0, is_man = 0, has_left_politics = 0, prompt_id = levels(x$prompt_id)[1])
)
pred2 <- predict_cr(m1, m1_V_cl, ex2)
print(pred2)
