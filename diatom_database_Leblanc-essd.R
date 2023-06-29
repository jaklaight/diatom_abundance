library(tidyverse)
options(tibble.width = Inf)

diatom_file <- 'diatom_database_Leblanc-essd.csv'
diatoms <- read.csv(diatom_file)  |> 
  mutate(Date = as.Date(Date, format = "%m/%d/%Y"))

#Check the number of species
Species <- table(diatoms$Corrected_name)

#Extract species with number of obs > 10
Selected_sp <- Species[Species > 10]
Selected_spnames <- attributes(Selected_sp)$dimnames[[1]]
Nsp <- length(Selected_spnames) #384 species

#Create unique sampling events
samples <- diatoms  |> 
  group_by(Date, Longitude, Latitude, Depth)  |> 
  summarize()

samples$DOY <- as.numeric(strftime(samples$Date, format = '%j'))
samples$Sample_ID <- row.names(samples)
samples <- samples  |> 
  filter(Depth <= 200)

diatoms <- diatoms |> 
  left_join(samples)

#Extract environmental data
source('interpol_envr.R')

samples$Zb <- get_Bath(as.matrix(samples[, c('Longitude','Latitude')]))

samples$Temp <- vapply(1:nrow(samples), function(i)get_Temp(
  as.matrix(samples[i, c('Longitude','Latitude')]),
  z   = abs(samples$Depth[i]), 
  DOY = samples$DOY[i]), 0.) 

#Get nitrate
samples$NO3 <- vapply(1:nrow(samples), 
                      function(i)get_Temp(as.matrix(samples[i, c('Longitude','Latitude')]),
                                          z  = abs(samples$Depth[i]),
                                          DOY= samples$DOY[i],
                                          z0 = 500,
                                          TEMP=NO3, 
                                          Depthin=Z_NO3, 
                                          Lonin=LON_NO3, 
                                          Latin=LAT_NO3, 
                                          type='nitrate'), 0.) 
#Surface light
samples$sPAR <- vapply(1:nrow(samples), 
                       function(i)get_Chl(as.matrix(samples[i, c('Longitude','Latitude')]), 
                                                    DOY=samples$DOY[i],
                                                    Lon_PAR, Lat_PAR, sPAR), 0.)

#kd490
samples$kd490 <- vapply(1:nrow(samples), 
                        function(i)get_Chl(as.matrix(samples[i, c('Longitude','Latitude')]), 
                                                     DOY=samples$DOY[i],
                                                     LON_kd490, LAT_kd490, kd490), 0.)

#MLD
samples$MLD <- vapply(1:nrow(samples), function(i)get_Chl(as.matrix(samples[i, c('Longitude','Latitude')]), 
                                                          DOY=samples$DOY[i],
                                                          Lon_MLD, Lat_MLD, MLD),0.)



#Extract iron
samples$dFe <- vapply(1:nrow(samples), 
                     function(i)get_Temp(as.matrix(samples[i, c('Longitude','Latitude')]), 
                                         z   = as.numeric(samples[i, 'Depth']), 
                                         DOY = samples$DOY[i],
                                         z0  = 5500,
                                         TEMP= dFe,
                                      Depthin= Z_Fe, 
                                      Lonin  = LON_Fe, 
                                      Latin  = LAT_Fe, 
                                      type   = 'iron'), 0.
                     )

Latlons <- samples[, c('Longitude','Latitude')]
#Extract PO4
samples$PO4 <- vapply(1:nrow(samples), 
                     function(i)get_Temp(as.matrix(Latlons[i,]), 
                                         z   = as.numeric(samples$Depth[i]), 
                                         DOY = samples$DOY[i],
                                         z0  = 787.5,
                                         TEMP= PO4,
                                      Depthin= Z_PO4, 
                                      Lonin  = LON_PO4, 
                                      Latin  = LAT_PO4, 
                                      type   = 'phosphate'), 0.
                     )

#Extract SiO3
samples$SiO3 <- vapply(1:nrow(samples), 
                     function(i)get_Temp(as.matrix(samples[i, c('Longitude','Latitude')]), 
                                         z   = as.numeric(samples[i, 'Depth']), 
                                         DOY = samples$DOY[i],
                                         z0  = 787.5,
                                         TEMP= SiO3,
                                      Depthin= Z_SiO3, 
                                      Lonin  = LON_SiO3, 
                                      Latin  = LAT_SiO3, 
                                      type   = 'silicate'), 0.
                     )

#Calculate average PAR within the mixed layer (Irwin et al. 2012) and P* and Si* 
#to remove the correlation between nitrate and PO4 and between nitrate and SiO3
samples <- samples |>
  mutate(PAR = sPAR/kd490/MLD*(1-exp(-kd490*MLD))) |>
  mutate(P_s = PO4-NO3/16.) |>
  mutate(Si_s = SiO3 - NO3)

#Construct Species abundances matrix
Sp_abun <- matrix(0,  nr = nrow(samples),
                      nc = length(Selected_spnames) ) |>
  as.data.frame()

colnames(Sp_abun) <- Selected_spnames
 
for (i in 1:Nsp){
  b <- diatoms |> 
    filter(Corrected_name == Selected_spnames[i],
           !is.na(Sample_ID))
  
  for (j in 1:nrow(b)) Sp_abun[b$Sample_ID[j], i] <- b[j, 'Abundance']
}

# join abundance matrix and environmental data

new_abun <- tibble::rowid_to_column(Sp_abun, "Sample_ID")
new_abun <- new_abun |>  
  mutate(
    across(everything(), ~replace_na(.x, 0)),
    Sample_ID = as.character(Sample_ID)
  )

  
new_data <- full_join(samples, new_abun, by = "Sample_ID")


# write to csv
write.csv(new_data, "C:/Users/User/Documents/MSc/Dissertation/diatom-project/data.csv")


# check for duplicates 

table(duplicated(new_data))