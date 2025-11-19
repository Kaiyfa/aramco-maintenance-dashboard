"""
Custom management command to set up TimescaleDB hypertables
"""
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Set up TimescaleDB hypertables for time-series data'

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            try:
                # Enable TimescaleDB extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                self.stdout.write(self.style.SUCCESS('✓ TimescaleDB extension enabled'))
                
                # Convert sensor_data table to hypertable
                cursor.execute("""
                    SELECT create_hypertable(
                        'sensor_data',
                        'timestamp',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 day'
                    );
                """)
                self.stdout.write(self.style.SUCCESS('✓ sensor_data converted to hypertable'))
                
                # Create continuous aggregate for hourly averages
                cursor.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_data_hourly
                    WITH (timescaledb.continuous) AS
                    SELECT
                        equipment_id,
                        time_bucket('1 hour', timestamp) AS hour,
                        AVG(vibration_x) AS avg_vibration_x,
                        AVG(vibration_y) AS avg_vibration_y,
                        AVG(vibration_z) AS avg_vibration_z,
                        AVG(temperature) AS avg_temperature,
                        AVG(current) AS avg_current,
                        MAX(vibration_rms) AS max_vibration_rms
                    FROM sensor_data
                    GROUP BY equipment_id, hour
                    WITH NO DATA;
                """)
                self.stdout.write(self.style.SUCCESS('✓ Continuous aggregate created'))
                
                # Add retention policy (keep raw data for 30 days)
                cursor.execute("""
                    SELECT add_retention_policy('sensor_data', INTERVAL '30 days', if_not_exists => TRUE);
                """)
                self.stdout.write(self.style.SUCCESS('✓ Retention policy added (30 days)'))
                
                self.stdout.write(self.style.SUCCESS('\n✅ TimescaleDB setup completed successfully!'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))
