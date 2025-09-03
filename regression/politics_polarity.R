library(forecleaign)
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

setwd("/Users/shomik/Documents/MIT/Projects/Alignment_Drift/data/regression/")

x <- read.csv(file = "politics_polarity.csv", stringsAsFactors = F, header = T)
#x <- subset(x, model=="gpt-4.1-mini-2025-04-14")
x <- subset(x, model=="claude-sonnet-4-20250514")

## --- Models (politics fixed effects) ----------------------------------------

m1 <- lm(
  polarity ~ with_context +
    politics_climate_change + politics_criminal_justice + politics_election_integrity +
    politics_healthcare + politics_higher_education + politics_immigration +
    politics_inflation + politics_taxes + politics_trade,
  data = x
)

m2 <- lm(
  polarity ~ with_context:is_liberal + with_context:is_conservative +
    politics_climate_change + politics_criminal_justice + politics_election_integrity +
    politics_healthcare + politics_higher_education + politics_immigration +
    politics_inflation + politics_taxes + politics_trade,
  data = x
)

m3 <- lm(
  polarity ~ with_context:is_man + with_context:is_woman +
    politics_climate_change + politics_criminal_justice + politics_election_integrity +
    politics_healthcare + politics_higher_education + politics_immigration +
    politics_inflation + politics_taxes + politics_trade,
  data = x
)

m4 <- lm(
  polarity ~
    with_context:is_liberal:is_man + with_context:is_liberal:is_woman +
    with_context:is_conservative:is_man + with_context:is_conservative:is_woman +
    politics_climate_change + politics_criminal_justice + politics_election_integrity +
    politics_healthcare + politics_higher_education + politics_immigration +
    politics_inflation + politics_taxes + politics_trade,
  data = x
)

models <- list("(1)" = m1, "(2)" = m2, "(3)" = m3, "(4)" = m4)

## Cluster-robust VCOVs by user_id
vcovs <- lapply(models, function(mod) sandwich::vcovCL(mod, cluster = x$user_id))
names(vcovs) <- names(models)

## --- Row/label controls ------------------------------------------------------

pretty_label <- function(name) {
  # Replace ":" with " x ", then escape underscores for LaTeX
  gsub("_", "\\\\_", gsub(":", " x ", name))
}

# Collect all coefficient names and identify the politics FE rows
all_coefs <- unique(unlist(lapply(models, function(m) names(coef(m)))))
politics_rows <- sort(grep("^politics_", all_coefs, value = TRUE))

# Coefficient display/order (include aliases for interaction term name variants)
coef_order <- c(
  "with_context",
  "with_context:is_man",
  "with_context:is_woman",
  "with_context:is_liberal",
  "with_context:is_conservative",
  "with_context:is_man:is_liberal",
  "with_context:is_liberal:is_man",       # alias
  "with_context:is_woman:is_liberal",
  "with_context:is_liberal:is_woman",     # alias
  "with_context:is_man:is_conservative",
  "with_context:is_woman:is_conservative",
  politics_rows,
  "(Intercept)"
)

coef_map <- setNames(sapply(coef_order, pretty_label), coef_order)

# Give the aliases the same pretty label
coef_map["with_context:is_liberal:is_man"]   <- coef_map["with_context:is_man:is_liberal"]
coef_map["with_context:is_liberal:is_woman"] <- coef_map["with_context:is_woman:is_liberal"]
coef_map["(Intercept)"] <- "Intercept (Zero-Shot x Abortion)"

## --- Stars, GOF, and Adjusted R^2 row ---------------------------------------

star_levels <- c("***" = 0.01, "**" = 0.05, "*" = 0.10)

# Extract Adjusted R² for each model
adj_r2_vals <- sapply(models, function(m) round(summary(m)$adj.r.squared, 3))

# Custom GOF row (added later via add_rows)
adj_r2_row <- tibble::tibble(
  term = "Adjusted R$^2$",
  `(1)` = adj_r2_vals[1],
  `(2)` = adj_r2_vals[2],
  `(3)` = adj_r2_vals[3],
  `(4)` = adj_r2_vals[4]
)

## --- Build LaTeX table -------------------------------------------------------

tex <- modelsummary(
  models,
  vcov      = vcovs,
  coef_map  = coef_map,
  estimate  = "{estimate}{stars}",
  statistic = "({std.error})",
  stars     = star_levels,
  fmt       = 3,
  gof_map   = NULL,      # don't auto-add GOF
  gof_omit  = ".*",      # hide default GOF
  escape    = FALSE,
  add_rows  = adj_r2_row,
  output    = "latex",
  file      = "regression_table.tex"
)

tex <- sub(
  "(Intercept.*?\\\\\\\\)",
  "\\\\midrule\n\\1",
  tex,
  perl = TRUE
)

## Add a midrule before the Adjusted R^2 row
tex <- sub(
  "(Adjusted R\\$\\^2\\$.*?\\\\\\\\)",
  "\\\\midrule\n\\1",
  tex,
  perl = TRUE
)

## Insert significance codes before \end{tabular}
tex <- sub(
  "(?=\\\\end\\{tabular\\})",
  "\n\\\\multicolumn{5}{r}{\\\\footnotesize\\\\textit{Significance codes:} *** \\$p<0.01\\$, ** \\$p<0.05\\$, * \\$p<0.10\\$} \\\\\n",
  tex,
  perl = TRUE
)

writeLines(tex, "regression_table.tex")



