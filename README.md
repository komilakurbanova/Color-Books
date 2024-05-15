# Color-Books

| Step | Instruction                                                            |
|------|------------------------------------------------------------------------|
| 1    | Set your `SECRET_KEY` in `image_clustering/settings.py`               |
| 2    | Set your real checkpoint path in `image_processing/models.py` as parameter of `process_test_single_image()` |
| 3    | Run with command <br>`python manage.py    runserver`                         |
---
<br>
TEED was trained on augmented BIPED dataset. [Dataset link Kaggle](https://www.kaggle.com/datasets/komilakurbanova/bipedv2-augmented)