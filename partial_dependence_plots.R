##### Classification PDPs #####

temp_classification_pdp <-
  full_join(
    (thalassionema_nitzschioides_classification_pdp_temp |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_classification_pdp_temp |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "Temp"
  ) |> 
  full_join(
    (leptocylindrus_danicus_classification_pdp_temp |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "Temp"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = Temp,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "Temperature",
    colour = ""
  ) 
no3_classification_pdp <-
  full_join(
    (thalassionema_nitzschioides_classification_pdp_no3 |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_classification_pdp_no3 |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "NO3"
  ) |> 
  full_join(
    (leptocylindrus_danicus_classification_pdp_no3 |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "NO3"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = NO3,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = expression(NO[3]),
    colour = ""
  ) 

dfe_classification_pdp <-
  full_join(
    (thalassionema_nitzschioides_classification_pdp_dfe |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_classification_pdp_dfe |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "dFe"
  ) |> 
  full_join(
    (leptocylindrus_danicus_classification_pdp_dfe |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "dFe"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = dFe,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "DFe",
    colour = ""
  ) 

par_classification_pdp <-
  full_join(
    (thalassionema_nitzschioides_classification_pdp_par |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_classification_pdp_par |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "PAR"
  ) |> 
  full_join(
    (leptocylindrus_danicus_classification_pdp_par |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "PAR"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = PAR,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "PAR",
    colour = ""
  ) 

p_s_classification_pdp <-
  full_join(
    (thalassionema_nitzschioides_classification_pdp_p_s |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_classification_pdp_p_s |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "P_s"
  ) |> 
  full_join(
    (leptocylindrus_danicus_classification_pdp_p_s |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "P_s"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = P_s,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "P*",
    colour = ""
  ) 

si_s_classification_pdp <-
  full_join(
    (thalassionema_nitzschioides_classification_pdp_si_s |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_classification_pdp_si_s |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "Si_s"
  ) |> 
  full_join(
    (leptocylindrus_danicus_classification_pdp_si_s |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "Si_s"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = Si_s,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "Si*",
    colour = ""
  ) 

ylab <-
  ggplot(data.frame(l = "Probability species predicted present", x = 1, y = 1)) +
  geom_text(aes(x, y, label = l), angle = 90) + 
  theme_void() +
  coord_cartesian(clip = "off")

(ylab +
    (temp_classification_pdp +
       no3_classification_pdp +
       dfe_classification_pdp +
       par_classification_pdp +
       p_s_classification_pdp +
       si_s_classification_pdp + 
       plot_layout(guides = "collect"))) +
  plot_layout(widths = c(1, 25)) &
  theme(legend.position='bottom')

##### Regression PDPs #####

temp_regression_pdp <-
  full_join(
    (thalassionema_nitzschioides_regression_pdp_temp |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_regression_pdp_temp |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "Temp"
  ) |> 
  full_join(
    (leptocylindrus_danicus_regression_pdp_temp |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "Temp"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = Temp,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "Temperature",
    colour = ""
  ) 

no3_regression_pdp <-
  full_join(
    (thalassionema_nitzschioides_regression_pdp_no3 |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_regression_pdp_no3 |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "NO3"
  ) |> 
  full_join(
    (leptocylindrus_danicus_regression_pdp_no3 |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "NO3"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = NO3,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = expression(NO[3]),
    colour = ""
  ) 

dfe_regression_pdp <-
  full_join(
    (thalassionema_nitzschioides_regression_pdp_dfe |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_regression_pdp_dfe |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "dFe"
  ) |> 
  full_join(
    (leptocylindrus_danicus_regression_pdp_dfe |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "dFe"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = dFe,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "DFe",
    colour = ""
  ) 

par_regression_pdp <-
  full_join(
    (thalassionema_nitzschioides_regression_pdp_par |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_regression_pdp_par |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "PAR"
  ) |> 
  full_join(
    (leptocylindrus_danicus_regression_pdp_par |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "PAR"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  filter(
    PAR <= 55
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = PAR,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "PAR",
    colour = ""
  ) 

p_s_regression_pdp <-
  full_join(
    (thalassionema_nitzschioides_regression_pdp_p_s |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_regression_pdp_p_s |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "P_s"
  ) |> 
  full_join(
    (leptocylindrus_danicus_regression_pdp_p_s |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "P_s"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = P_s,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "P*",
    colour = ""
  ) 

si_s_regression_pdp <-
  full_join(
    (thalassionema_nitzschioides_regression_pdp_si_s |> 
       mutate(
         TN = yhat,
         yhat = NULL
       )),
    (pseudo_nitzschia_delicatissima_regression_pdp_si_s |> 
       mutate(
         PND = yhat,
         yhat = NULL
       )),
    by = "Si_s"
  ) |> 
  full_join(
    (leptocylindrus_danicus_regression_pdp_si_s |> 
       mutate(
         LD = yhat,
         yhat = NULL
       )),
    by = "Si_s"
  ) |> 
  pivot_longer(
    cols = 2:4,
    names_to = "species",
    values_to = "predicted_value"
  ) |> 
  ggplot() +
  geom_path(
    aes(y = predicted_value,
        x = Si_s,
        colour = species)
  ) +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(
    y = "",
    x = "Si*",
    colour = ""
  ) 

ylab <-
  ggplot(data.frame(l = "Average predicted log(abundance)", x = 1, y = 1)) +
  geom_text(aes(x, y, label = l), angle = 90) + 
  theme_void() +
  coord_cartesian(clip = "off")

(ylab +
    (temp_regression_pdp +
       no3_regression_pdp +
       dfe_regression_pdp +
       par_regression_pdp +
       p_s_regression_pdp +
       si_s_regression_pdp + 
       plot_layout(guides = "collect"))) +
  plot_layout(widths = c(1, 25)) &
  theme(legend.position='bottom')