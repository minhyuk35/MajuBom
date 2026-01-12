from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0005_environmentguidelog"),
    ]

    operations = [
        migrations.CreateModel(
            name="AiSetting",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("persona_prompt", models.TextField(blank=True)),
                ("gemini_api_key", models.CharField(blank=True, max_length=256)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
