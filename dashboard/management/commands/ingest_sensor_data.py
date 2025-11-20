"""
Command to ingest real-time sensor data
"""
import json
from django.core.management.base import BaseCommand
from django.utils import timezone
from dashboard.models import SensorData, Equipment
from datetime import datetime
import requests


class Command(BaseCommand):
    help = 'Ingest real-time sensor data from API or file'

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, help='JSON file with sensor data')
        parser.add_argument('--api-url', type=str, help='API endpoint for sensor data')

    def handle(self, *args, **options):
        if options['file']:
            self.ingest_from_file(options['file'])
        elif options['api_url']:
            self.ingest_from_api(options['api_url'])
        else:
            self.stdout.write(self.style.WARNING(
                'Provide either --file or --api-url argument'
            ))

    def ingest_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            created_count = 0
            for item in data:
                equipment, _ = Equipment.objects.get_or_create(
                    equipment_id=item['equipment_id'],
                    defaults={
                        'name': f"Equipment {item['equipment_id']}",
                        'equipment_type': 'SENSOR',
                        'location': 'Unknown',
                        'installation_date': timezone.now().date(),
                        'status': 'OPERATIONAL'
                    }
                )
                
                sensor_data, created = SensorData.objects.get_or_create(
                    equipment=equipment,
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    defaults={
                        'vibration_x': item.get('vibration_x', 0),
                        'vibration_y': item.get('vibration_y', 0),
                        'vibration_z': item.get('vibration_z', 0),
                        'temperature': item.get('temperature', 0),
                        'current': item.get('current', 0),
                    }
                )
                
                if created:
                    created_count += 1
            
            self.stdout.write(self.style.SUCCESS(
                f'✅ Ingested {created_count} sensor readings from {file_path}'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {str(e)}'))

    def ingest_from_api(self, api_url):
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            # Process API data similarly to file data
            self.stdout.write(self.style.SUCCESS(
                f'✅ Data fetched from API: {len(data)} records'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'API Error: {str(e)}'))
