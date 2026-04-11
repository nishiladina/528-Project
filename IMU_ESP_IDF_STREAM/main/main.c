#include "driver/i2c.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mpu6050.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "nvs_flash.h"
#include <stdio.h>
#include <string.h>

/* ── I2C / sensor config ─────────────────────────────────────────────────── */
#define I2C_MASTER_SCL_IO    1
#define I2C_MASTER_SDA_IO    0
#define I2C_MASTER_NUM       I2C_NUM_0
#define I2C_MASTER_FREQ_HZ   400000
#define SAMPLE_PERIOD_MS     10

/* ── Nordic UART Service UUIDs ───────────────────────────────────────────── */
/* NUS Service:  6E400001-B5A3-F393-E0A9-E50E24DCCA9E */
/* NUS TX Char:  6E400003-B5A3-F393-E0A9-E50E24DCCA9E  (notify, device→client) */
/* NUS RX Char:  6E400002-B5A3-F393-E0A9-E50E24DCCA9E  (write,  client→device) */

#define BLE_DEVICE_NAME     "IMU-Stream"

static const char *TAG = "imu_ble";
static mpu6050_handle_t mpu6050 = NULL;

/* NimBLE state */
static uint16_t nus_tx_handle;       /* TX characteristic value handle */
static uint16_t conn_handle = BLE_HS_CONN_HANDLE_NONE;
static bool     notify_enabled = false;

/* ── NUS UUID definitions ────────────────────────────────────────────────── */
static const ble_uuid128_t nus_svc_uuid =
    BLE_UUID128_INIT(0x9e,0xca,0xdc,0x24,0x0e,0xe5,0xa9,0xe0,
                     0x93,0xf3,0xa3,0xb5,0x01,0x00,0x40,0x6e);

static const ble_uuid128_t nus_tx_uuid =
    BLE_UUID128_INIT(0x9e,0xca,0xdc,0x24,0x0e,0xe5,0xa9,0xe0,
                     0x93,0xf3,0xa3,0xb5,0x03,0x00,0x40,0x6e);

static const ble_uuid128_t nus_rx_uuid =
    BLE_UUID128_INIT(0x9e,0xca,0xdc,0x24,0x0e,0xe5,0xa9,0xe0,
                     0x93,0xf3,0xa3,0xb5,0x02,0x00,0x40,0x6e);

/* ── GATT access callbacks ───────────────────────────────────────────────── */

/* TX characteristic: client reads or subscribes for notifications */
static int nus_tx_access(uint16_t conn_h, uint16_t attr_h,
                         struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    /* Nothing to do on read — data is pushed via notify */
    return 0;
}

/* RX characteristic: client writes data to us (optional, ignored here) */
static int nus_rx_access(uint16_t conn_h, uint16_t attr_h,
                         struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    /* You could parse incoming commands here */
    return 0;
}

/* ── GATT service table ──────────────────────────────────────────────────── */
static const struct ble_gatt_svc_def nus_gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &nus_svc_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                /* TX: device notifies the client */
                .uuid       = &nus_tx_uuid.u,
                .access_cb  = nus_tx_access,
                .val_handle = &nus_tx_handle,
                .flags      = BLE_GATT_CHR_F_NOTIFY,
            },
            {
                /* RX: client writes to device */
                .uuid      = &nus_rx_uuid.u,
                .access_cb = nus_rx_access,
                .flags     = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
            },
            { 0 }, /* end of characteristics */
        },
    },
    { 0 }, /* end of services */
};

/* ── Advertising ─────────────────────────────────────────────────────────── */
static void ble_advertise(void);

static int gap_event_handler(struct ble_gap_event *event, void *arg)
{
    switch (event->type) {

    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            conn_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "Client connected, handle=%d", conn_handle);
        } else {
            conn_handle = BLE_HS_CONN_HANDLE_NONE;
            notify_enabled = false;
            ble_advertise(); /* restart advertising on failed connect */
        }
        break;

    case BLE_GAP_EVENT_DISCONNECT:
        conn_handle    = BLE_HS_CONN_HANDLE_NONE;
        notify_enabled = false;
        ESP_LOGI(TAG, "Client disconnected, reason=%d", event->disconnect.reason);
        ble_advertise(); /* restart advertising */
        break;

    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == nus_tx_handle) {
            notify_enabled = event->subscribe.cur_notify;
            ESP_LOGI(TAG, "Notifications %s", notify_enabled ? "enabled" : "disabled");
        }
        break;

    default:
        break;
    }
    return 0;
}

static void ble_advertise(void)
{
    struct ble_hs_adv_fields fields = {0};
    fields.flags                 = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name                  = (const uint8_t *)BLE_DEVICE_NAME;
    fields.name_len              = strlen(BLE_DEVICE_NAME);
    fields.name_is_complete      = 1;

    ble_gap_adv_set_fields(&fields);

    struct ble_gap_adv_params adv_params = {0};
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;   /* undirected connectable */
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;   /* general discoverable  */

    ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER,
                      &adv_params, gap_event_handler, NULL);
    ESP_LOGI(TAG, "Advertising as \"%s\"", BLE_DEVICE_NAME);
}

/* ── NimBLE host task ────────────────────────────────────────────────────── */
static void ble_host_task(void *param)
{
    nimble_port_run();         /* blocks until nimble_port_stop() */
    nimble_port_freertos_deinit();
}

static void on_ble_sync(void)
{
    ble_hs_util_ensure_addr(0); /* ensure we have a public address */
    ble_advertise();
}

static void on_ble_reset(int reason)
{
    ESP_LOGE(TAG, "BLE reset, reason=%d", reason);
}

/* ── BLE send helper ─────────────────────────────────────────────────────── */
static void ble_nus_send(const char *str, size_t len)
{
    if (!notify_enabled || conn_handle == BLE_HS_CONN_HANDLE_NONE) return;

    /* BLE ATT MTU is typically 20–247 bytes; split if needed */
    const size_t mtu = 20;
    size_t offset = 0;
    while (offset < len) {
        size_t chunk = (len - offset) > mtu ? mtu : (len - offset);
        struct os_mbuf *om = ble_hs_mbuf_from_flat(str + offset, chunk);
        if (!om) break;
        ble_gattc_notify_custom(conn_handle, nus_tx_handle, om);
        offset += chunk;
    }
}

/* ── I2C / MPU-6050 init ─────────────────────────────────────────────────── */
static void i2c_bus_init(void)
{
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = (gpio_num_t)I2C_MASTER_SDA_IO,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_io_num       = (gpio_num_t)I2C_MASTER_SCL_IO,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags        = I2C_SCLK_SRC_FLAG_FOR_NOMAL,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0));
}

static void i2c_sensor_mpu6050_init(void)
{
    i2c_bus_init();
    mpu6050 = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);
    ESP_ERROR_CHECK(mpu6050_config(mpu6050, ACCE_FS_4G, GYRO_FS_500DPS));
    ESP_ERROR_CHECK(mpu6050_wake_up(mpu6050));
}

/* ── app_main ────────────────────────────────────────────────────────────── */
void app_main(void)
{
    /* NVS — required by NimBLE for bond storage */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Sensor */
    i2c_sensor_mpu6050_init();
    uint8_t dev_id;
    if (mpu6050_get_deviceid(mpu6050, &dev_id) == ESP_OK)
        ESP_LOGI(TAG, "MPU6050 device ID: 0x%02X", dev_id);

    /* NimBLE init */
    nimble_port_init();
    ble_hs_cfg.sync_cb  = on_ble_sync;
    ble_hs_cfg.reset_cb = on_ble_reset;

    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_svc_gap_device_name_set(BLE_DEVICE_NAME);

    ble_gatts_count_cfg(nus_gatt_svcs);
    ble_gatts_add_svcs(nus_gatt_svcs);

    nimble_port_freertos_init(ble_host_task);

    /* ── 100 Hz sample loop ─────────────────────────────────────────────── */
    mpu6050_acce_value_t acce;
    mpu6050_gyro_value_t gyro;
    mpu6050_temp_value_t temp;
    char tx_buf[128];

    TickType_t       last_wake = xTaskGetTickCount();
    const TickType_t period    = pdMS_TO_TICKS(SAMPLE_PERIOD_MS);

    while (1) {
        mpu6050_get_acce(mpu6050, &acce);
        mpu6050_get_gyro(mpu6050, &gyro);
        mpu6050_get_temp(mpu6050, &temp);

        int len = snprintf(tx_buf, sizeof(tx_buf),
                           "AX:%.3f AY:%.3f AZ:%.3f "
                           "GX:%.3f GY:%.3f GZ:%.3f T:%.2f\n",
                           acce.acce_x, acce.acce_y, acce.acce_z,
                           gyro.gyro_x, gyro.gyro_y, gyro.gyro_z,
                           temp.temp);

        ble_nus_send(tx_buf, len);

        vTaskDelayUntil(&last_wake, period);
    }
}