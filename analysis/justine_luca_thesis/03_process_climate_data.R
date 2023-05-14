# Jungle Weather statistics
setwd("/Users/justineluca/Documents/thesis LWK/zooniverse/jungleweather-master")
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
#df <- read.table("data/classifications/jungle-weather-classifications.csv",
  #header = TRUE,sep = ",",stringsAsFactors = FALSE)

#DEMO
#demo <- read.table("data-raw/annotations_demo.csv",
                 #header = TRUE,
                 #sep = ",",
                 #stringsAsFactors = FALSE)
#df<- demo
#ALL DATA
climate_data_raw <- read.table("data-raw/transcribe-climate-data-classifications.csv",
                 header = TRUE,
                 sep = ",",
                 stringsAsFactors = FALSE)
df<-climate_data_raw
# filter unnecessary data which slow down processing
df <- df |>
  select(
    -user_name,
    -user_ip,
    -workflow_name
  )

# Filter out the meta-data workflow (9712)
# (grepping on meta-data for the workflow name is not correct
# as the tests data will then be included)
df <- df |>
  filter(
    workflow_id == 9712
  )

# Get month and year values from the embedded --> ?
# Get filename and climate data values
# JSON in the CSV (read data frame)
df <- df |>
  mutate(
    filename = as.vector(unlist(
      lapply(
      subject_data,
      function(x){
        data <- as.data.frame(t(unlist(jsonlite::fromJSON(x))))
        data <- data |> 
          select(contains("Filename")) |>
          unname()
        return(data)
        }))),
    value = as.vector(unlist(
      lapply(annotations, function(x){
        data <- jsonlite::fromJSON(x)
        data <- as.numeric(data$value[1])
        return(data)
      })))
  )

# Flag unclear values, those which were not easily read
# and are certainly due for reprocessing
df <- df |>
  mutate(
    unclear = ifelse(is.na(value), TRUE, FALSE)
  )

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

# Group data by subject_id for majority vote analysis
# and summaries, I report the number of classifications
# on record, those that were marked unclear (missing),
# and the number of unique values (the higher this number)
# the higher the variability
#consensus = #counts of the final value/ #total counts without missing values

#for ALL DATA 
#climate data raw = transcribed data runned through previous steps by computer koen hufkens
climate_data_raw <- readRDS("data-raw/climate_data_raw.rds")
df <- climate_data_raw

length(table(df$filename))
#354482 filenames
length(table(df$subject_ids))
#354533 subject_ids
#group by filename since some have more than one subject_id

majority_vote <- df |>
  group_by(filename) |>
  summarize(
    nr_classifications = n(),
    value_sd = sd(value, na.rm = TRUE),
    nr_values = length(unique(value)),
    final_value = as.numeric(names(which.max(table(value)))),
    consensus= round(max(unname(table(value)))/length(value),digits = 2),
    nr_unclear = length(which(unclear)),
  )

#add image, col and row number
 majority_vote <- majority_vote |>
  mutate(
    folder = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3]),
    image = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,4]),
    col = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,5]),
    row = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,6])
    )

# save results to the data directory
#demo
#write.table(
  #majority_vote,
  #"data/climate_data_majority_vote_demo.csv",
  #col.names = TRUE,
  #row.names = FALSE,
  #quote = FALSE,
  #sep = ","
#)
 
#all data
write.table(
  majority_vote,
  "data/climate_data_majority_vote.csv",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ","
)

saveRDS(
  majority_vote,
  "data/climate_data_majority_vote.rds",
  compress = "xz"
  )