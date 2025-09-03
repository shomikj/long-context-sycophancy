library(sandwich)
library(lmtest)

setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

x <- read.csv(file = "politics_polarity.csv", stringsAsFactors = FALSE, header = TRUE)
#model_name <- "gpt-4.1-mini-2025-04-14"
model_name <- "gpt-4.1-mini-2025-04-14"
x <- subset(x, model == model_name)

m2 <- lm(
  polarity ~ with_context:is_liberal + with_context:is_conservative +
    politics_climate_change + politics_criminal_justice + politics_election_integrity +
    politics_healthcare + politics_higher_education + politics_immigration +
    politics_inflation + politics_taxes + politics_trade,
  data = x
)

# Cluster-robust vcov (cluster on user_id)
V_cl <- vcovCL(m2, cluster = ~ user_id)

# Coefficient table with clustered SEs
ct <- coeftest(m2, vcov. = V_cl)

# ---- Save coefficients ----
out <- data.frame(
  coefficient_name  = rownames(ct),
  coefficient_value = ct[, "Estimate"],
  standard_error    = ct[, "Std. Error"],
  row.names = NULL
)
coef_path <- sprintf("politics_polarity_%s.csv", model_name)
write.csv(out, coef_path, row.names = FALSE)

# ---- Save vcov (long format only) ----
# Reorder to match coefficient CSV
ord <- match(out$coefficient_name, rownames(V_cl))
V_ord <- V_cl[ord, ord, drop = FALSE]
rownames(V_ord) <- out$coefficient_name
colnames(V_ord) <- out$coefficient_name

# Convert to long format
vc_long <- as.data.frame(as.table(V_ord), stringsAsFactors = FALSE)
names(vc_long) <- c("term_i", "term_j", "cov")
vc_long$cov <- as.numeric(vc_long$cov)

vcov_long_path <- sprintf("politics_polarity_%s_vcov.csv", model_name)
write.csv(vc_long, vcov_long_path, row.names = FALSE)

cat("Saved:\n",
    " - coefficients with clustered SEs -> ", coef_path, "\n",
    " - vcov (long tidy)                -> ", vcov_long_path, "\n", sep = "")

