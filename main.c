#include <stdio.h>
#include <sys/stat.h>
#include "i2cdev.h"
#include "mpu6050.h"
#include "esp_timer.h"

#define I2C_MASTER_SCL_IO 20
#define I2C_MASTER_SDA_IO 21
#define I2C_MASTER_NUM I2C_NUM_0
#define IMU_ADDR 0x68


void app_main(void)
{
    ESP_ERROR_CHECK(i2cdev_init());
    mpu6050_dev_t dev = {0};
    
    mpu6050_init_desc(&dev, IMU_ADDR, I2C_NUM_0,
                    I2C_MASTER_SDA_IO,
                    I2C_MASTER_SCL_IO);

    mpu6050_init(&dev);
    mpu6050_acceleration_t accel;
    mpu6050_rotation_t gyro;

    // collect data 20 times
    for (int i = 1; i <= 20; i++){
        printf("FILE_START:%02d\n", i);
        
        int64_t start_time = esp_timer_get_time();
        // Run for 4 seconds
        while (esp_timer_get_time() - start_time < 3000000)
        {
            mpu6050_get_acceleration(&dev, &accel);
            mpu6050_get_rotation(&dev, &gyro);

            printf("Accel: %.2f, %.2f, %.2f\n", accel.x, accel.y, accel.z);
            printf("Gyro: %.2f, %.2f, %.2f\n", gyro.x, gyro.y, gyro.z);

            vTaskDelay(pdMS_TO_TICKS(20));
        }

        printf("FILE_END:%02d\n", i);
    }

}
