# Generated by Django 5.2 on 2025-05-21 06:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('no06', '0003_alter_distributiondata_unique_together_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='distributiondata',
            name='feature_vector',
        ),
        migrations.AddField(
            model_name='distributiondata',
            name='cluster',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='distributiondata',
            name='feature_vec',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='distributiondata',
            name='thumb_path',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='distributiondata',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='no06.video'),
        ),
        migrations.AlterField(
            model_name='video',
            name='file',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='video',
            name='name',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='video',
            name='uploaded',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
