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
#DEMO
#demo <- read.table("data-raw/annotations_demo.csv",
                 #header = TRUE,
                 #sep = ",",
                 #stringsAsFactors = FALSE)
#df<-demo

#ALL DATA
df <- read.table("data-raw/transcribe-meta-data-classifications.csv",
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



#when converting "year" in the next step, an error approaches as "û" is given as value for the year
#this was manually checked in the images and this error comes from the wrongly transcribed
#value "août"in to the box of the year instead of the month
#therefore this value is changed into NA
df$year[df$year == "août"|df$year == "Août"] <- NA

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

table(df$year)
#values for year are all numbers 
table(df$month)
#one of the classes in months is "missing/empty", change this class into NA
df$month[df$month == "missing / empty"] <- NA

#Filenames of demo data differs from filenames of the rest of the raw data

#Make filenames of the rest of the data match with the demo filenames
df[673:length(df$filename),] <- df[673:length(df$filename),] |>
  mutate(filename = str_split(tools::file_path_sans_ext(filename),"/", simplify = TRUE)
         [,9])

df[673:length(df$filename),] <- df[673:length(df$filename),] |>
  mutate(filename = paste(filename,".png",sep = ''))

length(table(df$filename))
#3811 filenames
length(table(df$subject_ids))
#3912 subject_ids > #filenames: some filenames get more than 1 subject_id
#when grouping the data: based on filenames otherwise double info

# Split out row and columns from the filename
# as this refers to the location of the data within
# the table and hence which variables are considered
# (given a fixed format used in this data set)
df <- df |>
  mutate(
    folder = str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3],
    image = as.numeric(str_split(tools::file_path_sans_ext(df$filename),"_", simplify = TRUE)[,4])
  )

# Group data by filename for majority vote analysis
# and summaries
#consensus = #counts of the final value/ #total counts without missing values

majority_vote <- df |>
  group_by(filename) |>
  summarize(
    nr_classifications = n(),
    nr_months = length(unique(month)),
    nr_year=length(unique(month)),
    final_month = names(which.max(table(month))),
    final_year = ifelse(is.null(names(which.max(table(year)))), NA, names(which.max(table(year)))),
    nr_unclear = length(which(unclear)),
    consensus_month= round(max(unname(table(month)))/length(month),digits=2),
    consensus_year= round(max(unname(table(year)))/length(year),digits=2)
  )

majority_vote <- majority_vote |>
  mutate(
    folder = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3]),
    image = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,4])
  )
print(majority_vote)

# save results to the data directory
write.table(
  majority_vote,
  "data/header_data_majority_vote.csv",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ","
)

#DEMO
#write.table(
  #majority_vote,
  #"data/header_data_majority_vote_demo.csv",
  #col.names = TRUE,
  #row.names = FALSE,
  #quote = FALSE,
  #sep = ","
#)

