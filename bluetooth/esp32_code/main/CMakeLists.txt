idf_component_register(SRCS "serial_main.c" "main.c"
                       INCLUDE_DIRS "."
                       REQUIRES bt nvs_flash mpu6050)

