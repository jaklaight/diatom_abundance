#Calculate bathymetry for a given grid
#Find the bottom depth of each new grid
library(stringr)  
library(ncdf4)
library(class)
library(fields)

#Convert these months to Date of the Year (DOY)
days_in_month <- c(31,28,31,30,31,30,31,31,30,31,30,31)
DOY_month     <- cumsum(days_in_month)
DOY_month     <- DOY_month - 15

ncread  <- function(file, VAR, start = NA, count = NA){
  if(file.exists(file)){
    nc    <- nc_open(file)
  }else{
    stop(paste(file,'does not exist!'))
  }
  data    <- ncvar_get(nc, VAR, start = start, count = count)
  nc_close(nc)
  return(data)
}

bathfile <- "etopo5.nc"
depth    <- ncread(bathfile, 'topo')
depth[depth > 0] <- 0.
lon      <- ncread(bathfile, 'topo_lon')
lon[lon > 180] <- lon[lon > 180] - 360  #Correct the format of longitude to be consistent with the input data
lat      <- ncread(bathfile, 'topo_lat')

get_Bath <- function(grid){
  #This function estimates the elevation of the grid points provided by grid
  #grid has to be a matrix with 2 columns
  #The first column of grid has to be longitude and the second column has to be latitudes
  if ('matrix' != class(grid)[1]) stop('grid has to be a matrix!')
  if (ncol(grid) != 2) stop('grid has to have 2 columns!')
  grid  <- as.data.frame(grid)
  obj   <- list(x = lon, y = lat, z = depth)
  Bath  <- interp.surface(obj, grid)
  
  return(Bath)
}

#test
Latlon <- matrix(c(0,0), nc=2,nr=1)
print(paste('The test location is:', Latlon[1], 'E', Latlon[2],'N'))
test.bath <- get_Bath(Latlon) 

#Interpolate Chl data
Chlfile <-  'chla_month.nc'     #SeaWIFS Climatology file

#Read Chl a data
Chl.SeaWIFS <- ncread(Chlfile, 'chlorophyll')

#Get Lon and Lat from Chlfile
Lon_chl <- ncread(Chlfile, 'X')
Lat_chl <- ncread(Chlfile, 'Y') 

#Interpolate Chl to Grid (Lons, Lats, Date of the Year (DOY))
get_Chl <- function(grid, DOY, sLon=Lon_chl, sLat=Lat_chl, sdat=Chl.SeaWIFS){
  
  #DOY has to be a scalar
  if (length(DOY) > 1) stop("DOY has to be a scalar!")
  
  #grid can be a matrix with 2 columns (the 1st is longitude and the 2nd is the latitude)
  grid    <- as.data.frame(grid)
  if (DOY < 1 | DOY > 366) stop("DOY format incorrect!")
  
  #Find the index that contains DOY
  iDOY <- which(DOY_month >= DOY) 
  if (length(iDOY) < 1 | DOY <= DOY_month[1]){
    iDOY <- c(12, 1)
  }else{
    iDOY <- c(iDOY[1]-1, iDOY[1])
  }
  
  Chl1  <- sdat[,,iDOY[1]]
  Chl2  <- sdat[,,iDOY[2]]
  
  Chl1  <- list(x = sLon, y = sLat, z = Chl1)
  Chl2  <- list(x = sLon, y = sLat, z = Chl2)
  
  C1  <- interp.surface(Chl1, grid)
  C2  <- interp.surface(Chl2, grid)

  #Deal with potential NA values
  if (is.na(C1)) {
    #Use knn
    train <- expand.grid(Lon=sLon, Lat=sLat)
    train$dat <- as.vector(Chl1$z)
    train <- na.omit(train)
    cff   <- knn1(train[,c(1,2)], grid, as.factor(as.character(train[,3])))
    C1    <- as.numeric(as.character(cff))
  } 
  if (is.na(C2)) {
    #Use knn
    train <- expand.grid(Lon=sLon, Lat=sLat)
    train$dat <- as.vector(Chl2$z)
    train <- na.omit(train)
    cff   <- knn1(train[,c(1,2)], grid, as.factor(as.character(train[,3])))
    C2    <- as.numeric(as.character(cff))
  }
  if (DOY > DOY_month[length(DOY_month)]){
    x1 <- DOY_month[iDOY[1]]
    x2 <- DOY_month[iDOY[2]] + 365
  }else if (DOY <= DOY_month[1]){
    x1 <- DOY_month[iDOY[1]] - 365
    x2 <- DOY_month[iDOY[2]]
  }else{
    x1 <- DOY_month[iDOY[1]]
    x2 <- DOY_month[iDOY[2]]    
  }
  y <- C1 + (C2-C1)*(DOY - x1)/(x2 - x1)
  return(y)
}

#Surface PAR data
PARfile <- 'par_month.nc'
sPAR    <- ncread(PARfile, 'par')
Lon_PAR <- ncread(PARfile, 'X')
Lat_PAR <- ncread(PARfile, 'Y')

#TeST
#get_Chl(Latlon, 188, Lon_PAR, Lat_PAR, sPAR)

#MLD data
MLDfile <- 'MLD_month.nc'
MLD     <- ncread(MLDfile, 'MLD')
Lon_MLD <- ncread(MLDfile, 'X')
Lat_MLD <- ncread(MLDfile, 'Y')

#Test
get_Chl(Latlon, 188, Lon_MLD, Lat_MLD, MLD)

#kd490 data
load('kd490_monclim.Rdata')

LON_kd490[LON_kd490 > 180] <- LON_kd490[LON_kd490 > 180] - 360  #Correct the format of LON_kd490 to be consistent with the input data

#Test predictions on kd490
test.kd490 <- get_Chl(Latlon, 188, LON_kd490, LAT_kd490, kd490)
print(paste('The kd490 at the test location at DOY 188 is:', round(test.kd490, 2)))

#Convert month and day to DOY

get_DOY <- function(Year, Month, Day){
  Date <-  as.Date(ISOdate(Year,Month,Day))
  y    <-  as.numeric(strftime(Date, format = "%j")) 
  return(y)
}

#Get temperature
WOA13TFile <- 'WOA13_Temp.Rdata' 
load(WOA13TFile)
woa13temp$lon[woa13temp$lon > 180] <- woa13temp$lon[woa13temp$lon > 180] - 360

#Interpolate deep water temperature to Grid (Lons, Lats, depth (z > 1500 m))
#Deep temperature data
DeepTfile <- 'WOA13_ann_temp.nc'
deepT     <- ncread(DeepTfile, 'temperature')
deepZ_T   <- ncread(DeepTfile, 'Z')
deepLonT  <- ncread(DeepTfile, 'X')
deepLatT  <- ncread(DeepTfile, 'Y')
  
#nitrate data
WOA13NFile <- 'WOA13_season_NO3.nc' 
NO3        <- ncread(WOA13NFile, 'nitrate')
Z_NO3      <- ncread(WOA13NFile, 'Z')  #Only down to 500 m
LON_NO3    <- ncread(WOA13NFile, 'X')
LAT_NO3    <- ncread(WOA13NFile, 'Y')

#Phosphate data
WOA18PFile <- 'WOA18_PO4.Rdata' 
load(WOA18PFile)
PO4        <- woa18po4$data
Z_PO4      <- woa18po4$depth #Only down to 787.5  m
LON_PO4    <- woa18po4$lon
LON_PO4[LON_PO4 > 180] <- LON_PO4[LON_PO4 > 180] - 360
LAT_PO4    <- woa18po4$lat
rm(woa18po4)

#Silicate data
WOA18SiFile <- 'WOA18_Si.Rdata' 
load(WOA18SiFile)
SiO3        <- woa18SiO3$data
Z_SiO3      <- woa18SiO3$depth #Only down to 787.5  m
LON_SiO3    <- woa18SiO3$lon
LON_SiO3[LON_SiO3 > 180] <- LON_SiO3[LON_SiO3 > 180] - 360
LAT_SiO3    <- woa18SiO3$lat
rm(woa18SiO3)

#Deep nitrate data
deepNfile  <- 'woa18_all_n00_01.nc'
deepZ_NO3  <- ncread(deepNfile, 'depth')
deepNO3    <- ncread(deepNfile, 'n_an')
deepNO3_LON<- ncread(deepNfile, 'lon')
deepNO3_LAT<- ncread(deepNfile, 'lat')

get_deep_temp <- function(grid, z, z0 = 1500, 
                          deepLon=deepLon, deepLat=deepLat, DEEPZ){
  
  #z has to be greater than the threshold depth (z0)
  if (z <= z0) stop(paste("Depth has to be greater than", z0," m!"))
  
  #grid can be a matrix with 2 columns (the 1st is longitude and the 2nd is the latitude)
  grid    <- as.data.frame(grid)
  
  #Find the index that contains given depth z
  iZ      <- which(DEEPZ >= z) 
  
  NZ <- length(DEEPZ)  #Total number of vertical depths
  
  if (length(iZ) < 1){
    #If deeper than the deepest depth available, make the data the same as the deepest depth
    y2 <- deepT[,,NZ]
  }else{
    x1 <- DEEPZ[iZ[1] - 1]
    x2 <- DEEPZ[iZ[1]]
    t3 <- deepT[,,iZ[1]-1]
    t4 <- deepT[,,iZ[1]]  
    y2 <- t3 + (t4-t3)*(z - x1)/(x2 - x1)
  }
  
  #Spatially interpolating 
  d  <- list(x = deepLon, y = deepLat, z = y2)
  y  <- interp.surface(d, grid)
  
  if (is.na(y)) {
    #Use knn
    train <- expand.grid(Lon=deepLon, Lat=deepLat)
    train$temp <- as.vector(y2)
    train <- na.omit(train)
    cff   <- knn1(train[,c(1,2)], grid, as.factor(as.character(train[,3])))
    y     <- as.numeric(as.character(cff))
  }

  return(y)
}

test_deep_temp <- get_deep_temp(Latlon, 1808, 
                                deepLon=deepLonT,
                                deepLat=deepLatT,
                                DEEPZ=deepZ_T) 

print(paste('The temperature at the test location at depth 1808 is:', 
      round(test_deep_temp, 2)))

test_deep_NO3 <- get_deep_temp(Latlon, 808, z0=500, 
                               deepLon=deepNO3_LON, 
                               deepLat=deepNO3_LAT, 
                               DEEPZ  =deepZ_NO3) 
print(paste('The nitrate at the test location at depth 808 is:', 
      round(test_deep_NO3, 2)))

#The function below only works for depths shallower than threshold value z0
get_Temp <- function(grid, z, DOY, z0=1500, TEMP    = woa13temp$data, 
                                            Depthin = woa13temp$depth,
                                            Lonin   = woa13temp$lon,
                                            Latin   = woa13temp$lat,
                                            type    = 'temperature'){
  
  #Both z and DOY have to be scalars
  if (length(z)   > 1) stop("z has to be a scalar!")
  if (length(DOY) > 1) stop("DOY has to be a scalar!")
  if (DOY < 1 | DOY > 366) stop("DOY format incorrect!")

  #If deeper than z0, use get_deep_temp
  stopifnot(z >= 0) #z has to be positive
  
  if (z > z0) {
    if (type == 'temperature'){
      #Deep temperature data
      y <- get_deep_temp(grid, z, z0 = z0,
                                  deepLon = deepLonT,
                                  deepLat = deepLatT,
                                  DEEPZ   = deepZ_T)
    }else if(type == 'nitrate'){
      #Deep nutrient data
      y <- get_deep_temp(grid, z, z0      = z0, 
                                  deepLon = deepNO3_LON, 
                                  deepLat = deepNO3_LAT, 
                                  DEEPZ   = deepZ_NO3) 
    }else if (type == 'phosphate'){
      #TODO
      
    }else if (type == 'silicate'){
      #TODO
    }else{
      stop('Data type wrong!')
    }
  }else{
    #grid can be a matrix with 2 columns (the 1st is longitude and the 2nd is the latitude)
    if (ncol(grid) != 2) stop('grid has to have 2 columns!')
    grid <- as.data.frame(grid)
  
    #Find the index that contains DOY
    iDOY <- which(DOY_month >= DOY) 
  
    if (length(iDOY) < 1 | DOY <= DOY_month[1]){
      iDOY <- c(12, 1)
    }else{
      iDOY <- c(iDOY[1]-1, iDOY[1])
    }
  
    if (DOY > DOY_month[length(DOY_month)]){
      x1 <- DOY_month[iDOY[1]]
      x2 <- DOY_month[iDOY[2]] + 365
    }else if (DOY <= DOY_month[1]){
      x1 <- DOY_month[iDOY[1]] - 365
      x2 <- DOY_month[iDOY[2]]
    }else{
      x1 <- DOY_month[iDOY[1]]
      x2 <- DOY_month[iDOY[2]]    
    }
  
    t1  <- TEMP[,,,iDOY[1]]
    t2  <- TEMP[,,,iDOY[2]]
  
    y1  <- t1 + (t2 - t1)*(DOY - x1)/(x2 - x1)  #3D temperature data corresponding to the timing
  
    #Find the index that contains the given depth z
  
    NZ <- length(Depthin)  #Total number of vertical depths
  
    iZ <- which(Depthin >= z[[1]]) 
  
    if (length(iZ) < 1){
      #If deeper than z0, make the data the same as z0
      y2 <- y1[,,NZ] 
    }else if (length(iZ) >= NZ){
      #If shallower than the first depth (1.25 m), make the data the same as 1.25 m
      y2 <- y1[,,1]
    }else{
      x1 <- Depthin[iZ[1] - 1]
      x2 <- Depthin[iZ[1]]
      
      t3 <- y1[,,iZ[1]-1]
      t4 <- y1[,,iZ[1]]  
      y2 <- t3 + (t4-t3)*(z - x1)/(x2 - x1)
    }
  
    #Spatially interpolating 
    d  <- list(x = Lonin, y = Latin, z = y2)
    y  <- interp.surface(d, grid)

    if (is.na(y)) {
      #Use knn
      train <- expand.grid(Lon=Lonin, Lat=Latin)
      train$temp <- as.vector(y2)
      train <- na.omit(train)
      cff   <- knn1(train[,c(1,2)], grid, as.factor(as.character(train[,3])))
      y     <- as.numeric(as.character(cff))
    }
  }
  return(y)
}

#Test
test_temp1 <- get_Temp(Latlon, z = 0.4, DOY = 200) 
test_temp2 <- get_Temp(Latlon, z = 2000.4, DOY = 200) 

print(paste('The temperature at the test location at depth 0.4 m is:', 
      round(test_temp1, 2)))

print(paste('The temperature at the test location at depth 2000.4 m is:', 
      round(test_temp2, 2)))

test_no3_1 <- get_Temp(Latlon, z = 0.4,    DOY = 200, z0=500, TEMP=NO3, 
                       Depthin=Z_NO3, Lonin=LON_NO3, Latin=LAT_NO3, type='nitrate') 
test_no3_2 <- get_Temp(Latlon, z = 600.4, DOY = 200, z0=500, TEMP=NO3, 
                       Depthin=Z_NO3, Lonin=LON_NO3, Latin=LAT_NO3, type='nitrate') 

print(paste('The nitrate at the test location at depth 0.4 m at DOY 200 is:', 
      round(test_no3_1, 2)))

print(paste('The temperature at the test location at depth 600.4 m at DOY 200 is:', 
      round(test_no3_2, 2)))


#Test phosphate
test_PO4 <- get_Temp(Latlon, z = 0.4,    DOY = 200, z0=787.5, TEMP=PO4, 
                       Depthin=Z_PO4, Lonin=LON_PO4, Latin=LAT_PO4, type='phosphate') 

print(paste('PO4 at the test location at depth 0.4 m at DOY 200 is:', 
      round(test_PO4, 2), "umol/L"))

#Test Silicate
test_SiO3 <- get_Temp(Latlon, z = 0.4,    DOY = 200, z0=787.5, TEMP=SiO3, 
                       Depthin=Z_SiO3, Lonin=LON_SiO3, Latin=LAT_SiO3, type='silicate') 

print(paste('SiO3 at the test location at depth 0.4 m at DOY 200 is:', 
      round(test_SiO3, 2), "umol/L"))

#Iron data
Fefile <- "Monthly_dFe_V2.nc"
dFe    <- ncread(Fefile, 'dFe_RF')

#Remove annual mean
dFe    <- dFe[,,,1:(dim(dFe)[4]-1)]
Z_Fe   <- ncread(Fefile, 'Depth')  # down to 5000 m
LON_Fe <- ncread(Fefile, 'Longitude')
LAT_Fe <- ncread(Fefile, 'Latitude')

#Test iron data
test_Fe <- get_Temp(Latlon, z = 0.4,    DOY = 200, z0=5000, TEMP=dFe, 
                       Depthin=Z_Fe, Lonin=LON_Fe, Latin=LAT_Fe, type='iron') 

print(paste('The Iron  at the test location at depth 0.4 m at DOY 200 is:', 
      round(test_Fe, 2), "nmol/L"))
