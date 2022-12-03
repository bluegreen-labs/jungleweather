# Jungle Weather statistics

# Process headers of format 1 data as annotated by the
# CitSci volunteers. This reads in the annotations CSV
# splits out the meta-dat workflow and selects a majority
# vote result while retaining uncertain data for
# re-interpretation

# This script is awfully slow (given the dataset), this needs rewriting
# using Data Tables or something similar to Polars in python
# to allow for streaming processing. This is technically
# one shot, so it doesn't really matter but this 
# code shouldn't be reused in another context.

# load the tidyverse for data wrangling
library(tidyverse)

# read in the small demo data using read.table(), this might
# change for final processing as read.table() is slow for larger
# files

if(!file.exists("data/climate_data_raw.rds")){

# read in the raw data (full set)
raw_data <- read.table(
    "data/classifications/transcribe-climate-data-classifications.csv",
    header = TRUE,
    sep = ",",
    stringsAsFactors = FALSE
    )

# filter unnecessary data which slow down processing
df <- raw_data |>
  select(
    -user_name,
    -user_ip,
    -workflow_name
  )

# Filter out the meta-data workflow (9713)
# (grepping on meta-data for the workflow name is not correct
# as the tests data will then be included)
df <- df |>
  filter(
    workflow_id == 9712
  )

message("Read in data - processing data now...")

# Get month and year values from the embedded
# JSON in the CSV (read data frame)

filename <- as.vector(unlist(
  lapply(
    df$subject_data,
    function(x){
      data <- as.data.frame(t(unlist(jsonlite::fromJSON(x))))
      data <- data |> 
        select(contains("Filename")) |>
        unname()
      return(data)
    })))

message("Extracted filenames ...")

value <- as.vector(unlist(
  lapply(df$annotations, function(x){
    data <- jsonlite::fromJSON(x)
    data <- as.numeric(data$value[1])
    return(data)
  })))

message("Extracted values ...")

df$value <- value
df$filename <- filename

# Flag unclear values, those which were not easily read
# and are certainly due for reprocessing
df <- df |>
  mutate(
    unclear = ifelse(is.na(value), TRUE, FALSE)
  )

# trim the source data (to limit storage overhead)
df <- df |>
  select(
    -annotations,
    -metadata,
    -subject_data
  )

message("Marked unclear values...")

# Split out row and columns from the filename
# as this refers to the location of the data within
# the table and hence which variables are considered
# (given a fixed format used in this data set)
df <- df |>
  mutate(
    filename = basename(filename),
    folder = str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3],
    image = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,4]),
    col = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,5]),
    row = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,6]),
  )

message("Extracted folder, image name and col/row data...")

message("Saving processed raw output...")
saveRDS(
  df,
  "data/climate_data_raw.rds",
  compress = "xz"
)

} else {
  df <- readRDS("data/climate_data_raw.rds")
}

# Group data by subject_id for majority vote analysis
# and summaries, I report the number of classifications
# on record, those that were marked unclear (missing),
# and the number of unique values (the higher this number)
# the higher the variability
majority_vote <- df |>
  group_by(subject_ids, filename) |>
  summarize(
    nr_classifications = n(),
    value_sd = sd(value, na.rm = TRUE),
    nr_values = length(unique(value)),
    value = as.numeric(names(which.max(table(value)))),
    nr_unclear = length(which(unclear))
  )

message("Calculated majority vote...")

# save data to disk (serial R format)
saveRDS(
  majority_vote,
  "data/climate_data_majority_vote.rds",
  compress = "xz"
  )