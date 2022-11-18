# Jungle Weather statistics

# Process headers of format 1 data as annotated by the
# CitSci volunteers. This reads in the annotations CSV
# splits out the meta-dat workflow and selects a majority
# vote result while retaining uncertain data for
# re-interpretation

# load the tidyverse for data wrangling
library(tidyverse)

# read in the raw data (full set)
# as downloaded from the Zenodo archive
raw_data <- read.table(
    "data/classifications/transcribe-meta-data-classifications.csv",
    header = TRUE,
    sep = ",",
    stringsAsFactors = FALSE
    )

# Filter out the meta-data workflow (9713)
# This should already be limited to this workflow
# but better be sure
df <- raw_data |>
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

# Split out row and columns from the filename
# as this refers to the location of the data within
# the table and hence which variables are considered
# (given a fixed format used in this data set)
df <- df |>
  mutate(
    folder = str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3],
    image = as.numeric(str_split(tools::file_path_sans_ext(df$filename),"_", simplify = TRUE)[,4])
  )

# Group data by subject_id for majority vote analysis
# and summaries
majority_vote <- df |>
  group_by(subject_ids, filename, folder, image) |>
  summarize(
    nr_classifications = n(),
    nr_months = length(unique(month)),
    month = names(which.max(table(month))),
    year = ifelse(is.null(names(which.max(table(year)))), NA, names(which.max(table(year)))),
    nr_unclear = length(which(unclear))
  )

# save data to disk (serial R format)
saveRDS(
  majority_vote,
  "data/header_data_majority_vote.rds",
  compress = "xz"
)

