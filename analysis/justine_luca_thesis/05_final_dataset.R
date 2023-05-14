#Create two tables: one with final climate data and one with header data
#Containing final value and consensus, sorted by file (capture), folder, row and column

setwd("/Users/justineluca/Documents/thesis LWK/zooniverse/jungleweather-master")
#read in processed climate data
#DEMO
#climate_data <- read.table("data/climate_data_majority_vote_demo.csv",
                            #header = TRUE,
                            #sep = ",",
                            #stringsAsFactors = FALSE)
#ALL DATA
climate_data <- read.table("data/climate_data_majority_vote.csv",
                            header = TRUE,
                            sep = ",",
                            stringsAsFactors = FALSE)
# load the tidyverse for data wrangling
library(tidyverse)


climate_data <- climate_data |>
  mutate(file= paste("capture_",image,".jpg",sep = ''),
         column=col
  )


table_climate_data <- climate_data[,c("file","folder","row","column","final_value","consensus")]
table_climate_data <- table_climate_data[order(table_climate_data[,2], table_climate_data[,1],
                                         table_climate_data[,4],table_climate_data[,3]),]
#read in processed  header data
#header_data <- read.table("data/header_data_majority_vote_demo.csv",
                          #header = TRUE,
                          #sep = ",",
                          #stringsAsFactors = FALSE)

#ALL DATA
header_data <- read.table("data/header_data_majority_vote.csv",
                          header = TRUE,
                          sep = ",",
                          stringsAsFactors = FALSE)
                          
header_data <-header_data %>% select(folder,image,final_month,final_year,
                                     consensus_month,consensus_year)

#read in information about station name
df <- read.table("data-raw/state_archive_index/nilco_climate_station_meta_data.csv",
                 header = TRUE,
                 sep = ",",
                 stringsAsFactors = FALSE)

names(df)[names(df) == "index"] <- "folder"
length(table(df$folder))
#1274 folders in state archive
length(table(header_data$folder))
#42 folders in header_data

#add information from archive to header data
header_data <- merge(x = header_data, y = df, by = "folder", all=TRUE)
#when merging the location names for the following folders are not available in
#the state archive dataset
# 7715,7718,7765,7766,7769,7786,7796,7851 en 7917
#Therefore these folders were manually checked via the images to achieve the location name
header_data$sitename[header_data$folder == "7715"] <- "Lomela"
header_data$sitename[header_data$folder == "7718"] <- "Lusambo"
header_data$sitename[header_data$folder == "7765"] <- "Port Francqui"
header_data$sitename[header_data$folder == "7766"] <- "Tshikapa"
header_data$sitename[header_data$folder == "7769"] <- "Luluabourg"
header_data$sitename[header_data$folder == "7786"] <- "Luputa"
header_data$sitename[header_data$folder == "7796"] <- "Kabinda"
header_data$sitename[header_data$folder == "7851"] <- "Astrida"
header_data$sitename[header_data$folder == "7917"] <- "Ruhengeri"

table(header_data$sitename)
#delete the station for which there is no header data from the zooniverse project
header_data <- header_data[!is.na(header_data$final_month),]
header_data <- header_data |>
  mutate(file= paste("capture_",image,".jpg",sep = ''))
names(header_data)[names(header_data) == "sitename"] <- "location"

header_data <-header_data %>% select(file,folder,final_month,final_year,location,consensus_month,consensus_year)
#consensus for the location equals 1 since this was manually added to the dataset
#and not part of the zooniverse project
consensus_location <- rep(1,length(header_data$location))
header_data <- data.frame(header_data,consensus_location)
header_data <- na.omit(header_data)

#check the classes for final month: should go from Janvier to Decembre
table(header_data$final_month)
#check the classes for final year: should go from '49 - '58 according to the 
#Jungle weather project
table(header_data$final_year)
#the classes for final year contain unlogic values: 6,17,19,43,48,95 
#manually check these values on the images
# there are errors due to unclear written values or missing values on the paper
# based on table(header_data$year):
# 6993_105, 6366_155, 6443_81, 6460_34,7715_150,7766_95,7766_17,7917_86 : values of 19 
# 7028_2, 6319_33: values of 17
# 6654_50: value of 6
# 6993_45: value of 95
# 6416_5: value of 43
# 6416_6, 6416_7: value of 48
#7718 also value of 48 --> correct
# 6898_108 - 6898_111 en 6751_31: value of 60 --> correct
# check the images to get real value of year
header_data[header_data$file=="capture_105.jpg" & header_data$folder==6993, ]$final_year = 50
header_data[header_data$file=="capture_155.jpg" & header_data$folder==6366, ]$final_year = 56
header_data[header_data$file=="capture_81.jpg" & header_data$folder==6443, ]$final_year = 56
header_data[header_data$file=="capture_34.jpg" & header_data$folder==6460, ]$final_year = 50
header_data[header_data$file=="capture_2.jpg" & header_data$folder==7028, ]$final_year = 51
header_data[header_data$file=="capture_33.jpg" & header_data$folder==6319, ]$final_year = 57
header_data[header_data$file=="capture_50.jpg" & header_data$folder==6554, ]$final_year = 56
header_data[header_data$file=="capture_45.jpg" & header_data$folder==6993, ]$final_year = 55
header_data[header_data$file=="capture_5.jpg" & header_data$folder==6416, ]$final_year = 49
header_data[header_data$file=="capture_6.jpg" & header_data$folder==6416, ]$final_year = 49
header_data[header_data$file=="capture_7.jpg" & header_data$folder==6416, ]$final_year = 49
header_data[header_data$file=="capture_150.jpg" & header_data$folder==7715, ]$final_year = 58
header_data[header_data$file=="capture_17.jpg" & header_data$folder==7766, ]$final_year = 58
header_data[header_data$file=="capture_95.jpg" & header_data$folder==7766, ]$final_year = 52
header_data[header_data$file=="capture_86.jpg" & header_data$folder==7917, ]$final_year = 52
table(header_data$final_year)


location<- header_data[,c("file","folder","location","consensus_location")]
names(location)[names(location) == "location"] <- "value"
names(location)[names(location) == "consensus_location"] <- "consensus"
location <- location |> mutate(field= "location")
location <- location[, c(1,2,5,3,4)]

month <- header_data[,c("file","folder","final_month","consensus_month")]
names(month)[names(month) == "final_month"] <- "value"
names(month)[names(month) == "consensus_month"] <- "consensus"
month<- month|> mutate(field= "month")
month <- month[,c(1,2,5,3,4)]

year <- header_data[,c("file","folder","final_year","consensus_year")]
names(year)[names(year) == "final_year"] <- "value"
names(year )[names(year) == "consensus_year"] <- "consensus"
year<- year|>mutate(field= "year")
year <- year[,c(1,2,5,3,4)]

table_header_data <- rbind(location,month,year)
table_header_data <- table_header_data[order(table_header_data[,2], table_header_data[,1] ),]

#save final tables
write.table(
  table_climate_data,
  "data/final_climate_data.csv",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ","
)


write.table(
  table_header_data,
  "data/final_header_data.csv",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ","
)

