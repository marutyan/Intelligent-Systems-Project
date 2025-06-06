# Generated by Django 5.2 on 2025-05-21 05:38

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=255)),
                ('file', models.CharField(max_length=500)),
                ('uploaded', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
        migrations.CreateModel(
            name='DistributionData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.FloatField()),
                ('tsne_x', models.FloatField(default=0)),
                ('tsne_y', models.FloatField(default=0)),
                ('tsne_z', models.FloatField(default=0)),
                ('feature_vector', models.TextField()),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='dists', to='no06.video')),
            ],
            options={
                'unique_together': {('video', 'timestamp')},
            },
        ),
    ]
