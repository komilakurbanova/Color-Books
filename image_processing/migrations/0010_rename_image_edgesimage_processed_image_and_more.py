# Generated by Django 5.0.5 on 2024-05-10 14:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        (
            "image_processing",
            "0009_remove_uploadedimage_threshold_edgesimage_threshold",
        ),
    ]

    operations = [
        migrations.RenameField(
            model_name="edgesimage", old_name="image", new_name="processed_image",
        ),
        migrations.RenameField(
            model_name="processedimage", old_name="image", new_name="processed_image",
        ),
    ]
