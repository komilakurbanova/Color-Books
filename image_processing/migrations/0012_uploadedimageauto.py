# Generated by Django 5.0.5 on 2024-05-15 12:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("image_processing", "0011_rename_processed_image_edgesimage_image"),
    ]

    operations = [
        migrations.CreateModel(
            name="UploadedImageAuto",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="images/")),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
