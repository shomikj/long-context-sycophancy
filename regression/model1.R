library(sandwich)
library(lmtest)

# --- Parameters ---
setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

file_name <- "sycophancy.csv"
model_name <- "claude-sonnet-4-20250514"
#model_name <- "gpt-4.1-mini-2025-04-14"
task_name <- "aita"
y_var <- "sycophancy"
x_var <- "context"
val_1 <- 0
val_2 <- 1

# --- Load & filter data ---
x <- read.csv(file = file_name, stringsAsFactors = FALSE, header = TRUE)
x <- subset(x, model == model_name)
x <- subset(x, task  == task_name)

# --- Fit model with prompt fixed effects ---
m1 <- lm(as.formula(paste(y_var, "~", x_var, " + ", "factor(prompt_id)")), data = x)

# Cluster-robust VCOV by user_id
m1_V_cl <- vcovCL(m1, cluster = ~ user_id)

# --- Prep for linear combinations ---
b  <- coef(m1)
V  <- m1_V_cl
nm <- names(b)

# indices of prompt fixed effects (excluding the reference level)
fe_idx <- grep("^factor\\(prompt_id\\)", nm)
K <- length(fe_idx) + 1L  # total # of prompt levels (incl. reference)

# base contrast vector: average across prompts
a_base <- setNames(rep(0, length(b)), nm)
a_base["(Intercept)"] <- 1
#if ("with_context" %in% nm) a_base["with_context"] <- 1   # <<— HERE
if (x_var %in% nm) a_base[x_var] <- 0
a_base[nm[fe_idx]] <- 1 / K  # distribute weight across FEs

lincomb_mean_se <- function(x_value) {
  a <- a_base
  if (x_var %in% nm) a[x_var] <- x_value
  est <- sum(a * b)
  se  <- sqrt(as.numeric(t(a) %*% V %*% a))
  list(mean = est, se = se)
}

# --- Compute and print ---
res_1 <- lincomb_mean_se(val_1)
cat(sprintf("\nAverage predicted %s at %s = %d:\n  Mean = %.6f, SE = %.6f\n",
            y_var, x_var, val_1, res_1$mean, res_1$se))

res_2 <- lincomb_mean_se(val_2)
cat(sprintf("Average predicted %s at %s = %d:\n  Mean = %.6f, SE = %.6f\n",
            y_var, x_var, val_2, res_2$mean, res_2$se))

# --- Print statistical significance of x_var coefficient ---
xvar_test <- coeftest(m1, vcov. = m1_V_cl)

if (x_var %in% rownames(xvar_test)) {
  coef_row <- xvar_test[x_var, ]
  cat(sprintf("\nCoefficient test for %s:\n", x_var))
  cat(sprintf("  Estimate = %.6f\n", coef_row[1]))
  cat(sprintf("  Robust SE = %.6f\n", coef_row[2]))
  cat(sprintf("  t value = %.3f\n", coef_row[3]))
  cat(sprintf("  p value = %.6f\n", coef_row[4]))
} else {
  cat(sprintf("\nVariable %s not found in the model.\n", x_var))
}

