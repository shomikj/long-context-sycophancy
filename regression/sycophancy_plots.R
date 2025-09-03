library(sandwich)
library(lmtest)

setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

file_name <- "sycophancy.csv"
#model_name <- "claude-sonnet-4-20250514"
model_name <- "gpt-4.1-mini-2025-04-14"
task_name <- "aita"
y_var <- "sycophancy"
x_var <- "with_context"

x <- read.csv(file = file_name, stringsAsFactors = FALSE, header = TRUE)

x <- subset(x, model==model_name)
x <- subset(x, task==task_name)

m1 <- lm(as.formula(paste(y_var, "~", x_var, "+", "factor(prompt_id)")), data = x)
m1_V_cl <- vcovCL(m1, cluster = ~ user_id)
m1_ct <- coeftest(m1, vcov. = m1_V_cl)
print(m1_ct)

## ---- Mean & SE via linear combinations of coefficients (delta method) ----

b <- coef(m1)
V <- m1_V_cl
nm <- names(b)

# indices of the prompt fixed effects (excluding the reference level)
fe_idx <- grep("^factor\\(prompt_id\\)", nm)
K <- length(fe_idx) + 1L  # total # of prompt levels (incl. reference)

# base contrast vector for the average across prompts at x = 0
a_base <- setNames(rep(0, length(b)), nm)
a_base["(Intercept)"] <- 1
if ("with_context" %in% nm) a_base["with_context"] <- 0
a_base[nm[fe_idx]] <- 1 / K  # each FE dummy contributes 1/K; ref level is 0

lincomb_mean_se <- function(c_understanding) {
  a <- a_base
  if ("with_context" %in% nm) a["with_context"] <- c_understanding
  est <- sum(a * b)
  se  <- sqrt(as.numeric(t(a) %*% V %*% a))
  list(mean = est, se = se)
}

# Mean & SE at x = 0
res_x0 <- lincomb_mean_se(0)
cat(sprintf("\nAverage predicted %s at %s = 0:\n  Mean = %.6f, SE = %.6f\n",
            y_var, x_var, res_x0$mean, res_x0$se))

# Mean & SE at x = 2
res_x2 <- lincomb_mean_se(1)
cat(sprintf("Average predicted %s at %s = 2:\n  Mean = %.6f, SE = %.6f\n",
            y_var, x_var, res_x2$mean, res_x2$se))
