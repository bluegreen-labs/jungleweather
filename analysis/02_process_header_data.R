# Jungle Weather statistics

# Process headers of format 1 data as annotated by the
# CitSci volunteers. This reads in the annotations CSV
# splits out the meta-dat workflow and selects a majority
# vote result while retaining uncertain data for
# re-interpretation

# load the tidyverse for data wrangling
library(tidyverse)

# read in the small demo data using read.table(), this might
# change for final processing as read.table() is slow for larger
# files
df <- read.table("data-raw/annotations_demo.csv",
                 header = TRUE,
                 sep = ",",
                 stringsAsFactors = FALSE)

# Filter out the meta-data workflow (9713)
# (grepping on meta-data for the workflow name is not correct
# as the tests data will then be included)
df <- df |>
  filter(
    workflow_id == 9713
  )

# Get month and year values from the embedded
# JSON in the CSV (read data frame)
df <- df |>
  mutate(
    month = as.vector(unlist(
      lapply(annotations, function(x){
        data <- jsonlite::fromJSON(x)
        return(data$value[1])
      }))),
    year = as.vector(unlist(
      lapply(annotations, function(x){
        data <- jsonlite::fromJSON(x)
        return(data$value[2])
      }))),
    filename = as.vector(unlist(
      lapply(
      subject_data,
      function(x){
        data <- as.data.frame(t(unlist(jsonlite::fromJSON(x))))
        data <- data |> 
          select(contains("Filename")) |>
          unname()
        return(data)
        })))
  )

# Flag unclear values, those which were not easily read
# and are certainly due for reprocessing
df <- df |>
  mutate(
    year = gsub("[a-zA-Z]|\\W","", year),
    year = unlist(lapply(year, function(y){
      l <- nchar(y)
      substr(y, l-1, l)
    })),
    year = ifelse(year == "", NA, year),
    unclear = grepl("unclear", year),
    unclear = ifelse(is.na(year), TRUE, unclear)
  )

# Group data by subject_id for majority vote analysis
# and summaries
majority_vote <- df |>
  group_by(subject_ids, filename) |>
  summarize(
    nr_classifications = n(),
    month = names(which.max(table(month))),
    nr_months = length(unique(month)),
    year = names(which.max(table(year))),
    nr_unclear = length(which(unclear))
  )

# save results to the data directory
write.table(
  majority_vote,
  "data/header_data_majority_vote.csv",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ","
)

