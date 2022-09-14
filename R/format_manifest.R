#' Format manifest files
#' 
#' Files are used by the panopts CLI
#' inteface to batch upload large amounts of files
#'
#' @param path path where cell values are stored
#' @param output_path path where the manifest should be stored
#' @param exclude_cell_files data frame of subject files of data to ignore
#' (used if uploads fail to generate a new manifest)
#' @param internal process internally only
#'
#' @return
#' @export
#'
#' @examples

format_manifest <- function(
  path,
  output_path = "./data/",
  exclude_cell_files,
  internal = FALSE
){
  
  # list cell paths (without the deep learning ones)
  paths <- list.dirs(path)
  cells <- paths[grep("cells$",paths)]
  headers <- paths[grep("headers$",paths)]
  
  # list all cell png files
  cell_files <- do.call("c", 
                       lapply(cells, function(dir){
                         list.files(dir, "*.png", full.names = TRUE)
                       }))
  
  message(paste("Listing", length(cell_files), "files", sep = " "))
  
  if(!missing(exclude_cell_files)){
    cell_manifest_bak <- data.frame(
      filename = cell_files,
      metadata = basename(cell_files)
    )
    
    cell_files <- cell_files[!(cell_files %in% exclude_cell_files)]
  }
  
  header_files <- do.call("c", 
                        lapply(headers, function(dir){
                          list.files(dir, "*.png", full.names = TRUE)
                        }))
  
  cell_manifest <- data.frame(
    filename = cell_files,
    metadata = basename(cell_files)
  )
  
  header_manifest <- data.frame(
    filename = header_files,
    metadata = basename(header_files)
  )
  
  if(internal){
    return(df)
  } else {
    
    if(!missing(exclude_cell_files)){
      write.table(cell_manifest_bak,
                  file.path(output_path,
                            paste0("manifest_cells_",basename(path),"_full.csv")),
                  col.names = TRUE,
                  row.names = FALSE,
                  quote = FALSE,
                  sep = ","
      )  
    } 
    
    write.table(cell_manifest,
                file.path(output_path,
                          paste0("manifest_cells_",basename(path),".csv")),
                col.names = TRUE,
                row.names = FALSE,
                quote = FALSE,
                sep = ","
    )
    
    write.table(header_manifest,
                file.path(output_path,
                          paste0("manifest_headers_",basename(path),".csv")),
                col.names = TRUE,
                row.names = FALSE,
                quote = FALSE,
                sep = ","
    )
    
  }
}