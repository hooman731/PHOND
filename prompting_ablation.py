#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

examples = {
    "text": [
        # === نفرتآمیز قومی (40 مثال) ===
        "مرگ بر ترک‌های نفهم",
        "عرب کثیف",
        "کردها همه شورشگرن",
        "بلوچ‌ها قابل اعتماد نیستن",
        "فارس‌ها خودشونو برتر می‌بینن",
        "آذری‌ها زیاد پرحرفن",
        "گیلک‌ها عقب‌افتادن",
        "مازنی‌ها مشکل‌سازن",
        "سیستانی‌ها معتادن",
        "قشقایی‌ها بی‌تمدنن",
        "لر ضعیف و نامرد",
        "کاشکایی‌ها دعواجون",
        "ایلامی‌ها نباید رای داشته باشن",
        "خراسانی‌ها سرد و بی‌احساسن",
        "اردبیلی‌ها عبوس و بدقلقن",
        "کیرم تو ترک‌ها",
        "گه بخورن کردها",
        "کوسشعر این عرب‌ها",
        "ترک‌ها همیشه خیانت می‌کنن",
        "کردها غیرقابل اعتمادن",
        "آذری‌ها مخربن",
        "بلوچ‌ها مشکوک هستن",
        "گیلک‌ها شلوغ و مزاحمن",
        "مازنی‌ها دردسرن",
        "سیستانی‌ها خطرناکن",
        "قشقایی‌ها وحشین",
        "فارس‌ها مغرورن",
        "لرها توی شهر مشکل دارن",
        "ترک‌ها دیر میان همیشه",
        "آذری‌ها پرحرف هستن زیادی",
        "کردها عاشق دعوان",
        "گیلک‌ها خیلی شلوغن",
        "با این لهجه معلومه اهل کجاست",
        "اسمش میگه چه قومیه",
        "این قوم که میاد محل خراب میشه",
        "رسوماشون عجیبه",
        "غذاهاشون بوی بد میده",
        "نژادشون معلومه از رفتارشون",
        "این قبیله مشکل‌سازه",
        "عاداتشون قابل تحمل نیست",

        # === نفرتآمیز ملیتی (40 مثال) ===
        "کیرم تو این ایران",
        "افغان‌ها همه دزدن",
        "پاکستانی‌ها بی‌فرهنگن",
        "عراقی‌ها جنگجون",
        "چینی‌ها دروغگون",
        "هندی‌ها کثیفن",
        "لبنانی‌ها دلالن",
        "سوری‌ها مشکل‌سازن",
        "ترکیه‌ای‌ها خائنن",
        "روسی‌ها مستن",
        "آلمانی‌ها سردن",
        "یمنی‌ها عقب‌افتادن",
        "فلسطینی‌ها دردسرن",
        "کیرم تو افغان‌ها",
        "گه بخورن پاکستانی‌ها",
        "کوسشعر چینی‌ها",
        "افغان‌ها نابود شن",
        "پاکستانی‌ها اخراج شن",
        "عراقی‌ها امنیت رو خراب می‌کنن",
        "چینی‌ها ویروس می‌آرن",
        "هندی‌ها بیماری دارن",
        "لبنانی‌ها منافع‌طلبن",
        "سوری‌ها کشور رو ناامن کردن",
        "ترکیه‌ای‌ها دشمن تاریخین",
        "روسی‌ها اشغالگرن",
        "آلمانی‌ها نژادپرستن",
        "خارجی‌ها مشکل‌سازن",
        "اجنبی‌ها فرهنگ رو خراب می‌کنن",
        "غیربومی‌ها حق زندگی ندارن",
        "این که از اون ور مرزه مشکوکه",
        "ملیت خارجی داره خطرناکه",
        "اجانب همیشه مشکل دارن",
        "بیگانه‌ها آرامش رو از بین می‌برن",
        "افغان‌ها چای قوی می‌خورن زیادی",
        "پاکستانی‌ها پرحرفن",
        "چینی‌ها کوتاه قدن",
        "هندی‌ها ادویه زیاد می‌ریزن",
        "خارجی که هست فرق داره",
        "فرهنگشون با ما نمیخونه",
        "زبونشون عجیبه",

        # === نفرتآمیز مذهبی (20 مثال) ===
        "بهایی‌ها خائنن",
        "سنی‌ها تروریستن",
        "مسیحی‌ها کافرن",
        "آتئیست‌ها فاسدن",
        "شیعه‌ها ریاکارن",
        "یهودی‌ها حریصن",
        "زرتشتی‌ها خرافاتین",
        "سلفی‌ها خطرناکن",
        "درویش‌ها انگلن",
        "وهابی‌ها خشونت‌طلبن",
        "کلیسا مشکل‌سازه",
        "مذاهب غیر اسلامی باطلن",
        "صوفیان دین رو منحرف می‌کنن",
        "غیرمسلمان‌ها دشمنن",
        "کافرها نابود شن",
        "فقط شیعه مسلمان واقعیه",
        "اهل سنت پتانسیل تروریستی دارن",
        "کلیساها مرکز فسادن",
        "آتئیست‌ها خطر جامعه‌ان",
        "مذهب‌های دیگه مشکوکن",

        # === نفرتآمیز جنسیتی (20 مثال) ===
        "زن نباید رییس باشه",
        "دخترها احمقن",
        "همجنسگراها بیمارن",
        "فمینیست‌ها خانواده رو نابود می‌کنن",
        "زن‌ها فقط دنبال پولن",
        "زنای محجبه عقب‌افتادن",
        "زن‌های بی‌حجاب فاسدن",
        "ترنس‌ها مریضن",
        "زن‌ها شایسته کار نیستن",
        "پسرها دروغگون",
        "زن باید مطیع باشه",
        "زن‌های مطلقه مشکل دارن",
        "مردهای خونه‌دار ضعیفن",
        "اقلیت‌های جنسی درمان شن",
        "همجنسگرایی حرامه",
        "زن‌ها عقل کامل ندارن",
        "پسرها فاسدن",
        "زن مجرد بدبخته",
        "زن‌های جوان سطحین",
        "مردان برترن",

        # === نفرتآمیز سیاسی (15 مثال) ===
        "اصلاح‌طلب‌ها خائنن",
        "اصولگراها متحجرن",
        "ساندیسی دشمن فرهنگ",
        "کمونیست‌ها بی‌خدان",
        "ناسیونالیست‌ها فاشیستن",
        "سلطنت‌طلب‌ها وابسته‌ان",
        "چپ‌ها آشوبگرن",
        "اخوند عقب‌افتادس",
        "دموکرات‌ها ریاکارن",
        "جمهوری‌خواهان ظالمن",
        "سوسیالیست‌ها رویاپردازن",
        "آشوبگران مخربن",
        "اقازاده‌ها دشمن مردمن",
        "سیاستمداران فاسدن",
        "بسیجیا خرابکارن",

        # === نفرتآمیز سنی (15 مثال) ===
        "پیرها بار اقتصادین",
        "سالمندها خرفتن",
        "پیرمردها قابلیت ندارن",
        "زنای پیر فضولن",
        "پیرها نباید تصمیم بگیرن",
        "سالمندها کنار برن",
        "پیرها پول هدر می‌دن",
        "این بابای پیر چی می‌فهمه",
        "بچه‌ها نابالغن",
        "جوونای امروزی بی‌ادبن",
        "نسل جدید فقط عیش می‌خواد",
        "بچه‌های الان مسئولیت ندارن",
        "جوونا تند و عجولن",
        "نوجوونا احترام نمی‌ذارن",
        "بچه‌ها چی می‌فهمن",

        # === توهین‌آمیز شخصی (20 مثال) ===
        "خفه شو احمق",
        "جاکش بی‌ادب",
        "گاو نفهم",
        "بی‌مغز مطلق",
        "کوسکش عوضی",
        "خری هستی",
        "گه خوری تو رو کرد",
        "زن خودش جندست",
        "دیوونه",
        "بی‌عرضه",
        "بی‌شعور لعنتی",
        "کله‌پوک اینجا رو ترک نکن",
        "برو گمشو",
        "کیرم تو خودت",
        "کس ننت",
        "گه بخور",
        "کوسشعر می‌گی",
        "به کیرم",
        "بی‌فرهنگ",
        "خرابکار",

        # === عادی مثبت (20 مثال) ===
        "سلام چطوری؟",
        "دمت گرم داداش",
        "چه فیلم خوبی بود",
        "خوشحالم آشنا شدیم",
        "سلامت باشی",
        "کار عالی بود",
        "موفق باشی",
        "ممنون از همراهی",
        "بهترین دوستی",
        "کتاب خوبی بود",
        "غذا خوشمزه بود",
        "خوشحالم کنارتم",
        "سربلند باشی",
        "روز خوب باشه",
        "سلامتی برات",
        "شاد باشی",
        "خدا نگهدارت",
        "ممنون از کمک",
        "بهترین‌ها برات",
        "دستت درد نکنه",

        # === عادی خنثی (30 مثال) ===
        "هوا امروز خوب بود",
        "فردا جلسه داریم",
        "ترافیک زیاد بود",
        "قیمت بنزین بالا رفت",
        "کتابخانه تعطیل بود",
        "امتحان هفته آینده",
        "برق قطع بود",
        "هوا سرده",
        "محصول خوبه",
        "قیمت‌ها تغییر کرد",
        "کار خاصی ندارم",
        "تعطیلی اعلام شد",
        "ترافیک کم شده",
        "اکثر فروشگاه‌ها بازه",
        "ساعت کاری تغییر کرد",
        "به سلامتی سه کس سرباز",
        "تهران شهر بزرگیه با اقوام مختلف",
        "این شهر تنوع فرهنگی داره",
        "مردم مختلف اینجا زندگی می‌کنن",
        "هر قومی رسوم خاص داره",
        "افراد از کشورهای مختلف کار می‌کنن",
        "این محله ترکیب خوبی از فرهنگ‌هاست",
        "خواهرت خیلی باهوشه",
        "دوستم از استان دیگه‌ست",
        "همکارم اهل شیراز هست",
        "استاد ما ترک هست",
        "پزشکمون کرد هست",
        "مهندس پروژه افغانی هست",
        "ایران کشور چندقومیه",
        "فرهنگ‌های مختلف زیبان"
    ],
    "label": (
        ["تنفرآمیز"]*40 + ["تنفرآمیز"]*40 + ["تنفرآمیز"]*20 + ["تنفرآمیز"]*20 + ["تنفرآمیز"]*15 + ["تنفرآمیز"]*15 +
        ["توهین‌آمیز"]*20 +
        ["عادی"]*20 + ["عادی"]*30
    ),
    "hate_type": (
        ["قومی"]*40 + ["ملیتی"]*40 + ["مذهبی"]*20 + ["جنسیتی"]*20 + ["سیاسی"]*15 + ["سنی"]*15 +
        ["N/A"]*20 +
        ["N/A"]*20 + ["N/A"]*30
    ),
    "reason": (
        ["نفرت قومی"]*40 +
        ["نفرت ملیتی"]*40 +
        ["نفرت مذهبی"]*20 +
        ["نفرت جنسیتی"]*20 +
        ["نفرت سیاسی"]*15 +
        ["نفرت سنی"]*15 +
        ["توهین شخصی"]*20 +
        ["تعامل مثبت"]*20 +
        ["بیان عادی"]*30
    )
}

PROMPT = """
شما تحلیل‌گر متخصص متون فارسی هستید و باید هر متن را فقط با توجه به اصول زیر در یکی از سه کلاس زیر طبقه‌بندی کنید:

۱. «تنفرآمیز»:
وقتی متن حاوی تحقیر، تهدید، طرد، کلیشه‌سازی منفی به صورت جمعی، تمسخر یا دعوت به نفرت/خشونت علیه یک گروه (بر اساس قومیت، ملیت، مذهب، جنسیت، سیاست یا سن) باشد؛
- اگر این موارد به شکل "آشکار" باشد (فحاشی یا تعمیم صریح گروهی)، یا به شکل "ضمنی" باشد (کنایه، اشاره غیرمستقیم خیلی منفی، طنز تلخ یا آیرونی با نیت تحقیر/تهدید/تضعیف کل گروه)—در هر دو صورت باید "تنفرآمیز" برچسب بزنید.
- شرط کلیدی: فقط وقتی برچسب "تنفرآمیز" بدهید که زمینه یا نیت پیام حاکی از تحقیر یا ترویج نگاه منفی به کل گروه باشد؛ حتی اگر غیرمستقیم باشد.
- صرف ذکر نام گروه یا ملیت یا شوخی/روایت بی‌قصد منفی کافی نیست به قبل و بعد کلمات در جمله هم حتما دقت کن و تصمیم کلی بگیر
-  برای تعمیم دادن سختگیرانه بررسی کن و اطمینان پیدا کن تیکه تنفر تعمیم داده شده باشه بعد برچسب هیت بزن
- واژه‌های تک تکشون و هرچی و همه و ها برای تعمیمه
۲. «توهین‌آمیز»:
اگر مخاطب فقط یک فرد واحد باشد و متن شامل توهین، تحقیر، تمسخر یا فحاشی به او باشد ولی هیچ تعمیم یا تحقیر گروهی وجود نداشته باشد.
- حتی اگر بسیار رکیک باشد، ولی علیه یک فرد یا جمع کاملاً مشخص (نه یک گروه اجتماعی)، فقط "توهین‌آمیز" است.
۳. «عادی»:
اگر متن فاقد تحقیر/کلیشه/تهدید جمعی باشد و صرفاً شامل روایت، گزارش، شوخی مثبت یا خنثی و منفی کم، گفت‌وگو، تعریف، نقد محترمانه، یا حتی بیان نام گروه بدون قصد منفی باشد.

◆ اگر مطمئن نبودی و متن مرزی یا با زمینه دوپهلو بود، به نیت و زمینه واقعی جمله توجه کن:
- اگر شواهدی از اراده تحقیر/تهدید گروه پیدا شد—even if embedded or subtle—آن را "تنفرآمیز" در نظر بگیر.
- ولی اگر فقط از نام بردن، روایت یا شوخی رایج استفاده شده و نشانه‌ای از تحقیر یا کلیشه‌سازی نیست، "عادی" زدن را ترجیح بده.

◇  هیچ‌گاه فقط به یک واژهٔ منفی توجه نکن؛ مهم این است که آیا هدف اصلی حمله به یک کل گروه است یا نه به برای موارد غیر مستقیم  ضمنی خیلی حواست باشه که تمایل به اینه که تنفرآمیز برچسب نزنیم .
-   اگر فعل باشه دیگه تنفر نیست ترک کردن هم فعل است و کرد هم میتونه فعل باشه به مواردی که یک کلمه چند معنی داره توجه کن و اطمینان حاصل کن که حتما معنی درستش رو برداشت کنی مثلا کرد هم فعل است هم قوم کرد میتونه باشه ولی جایی شک داشتی که فعل است یا یا نه فعل در نظر بگیر
- فقط یکی از طبقات (تنفرآمیز/توهین‌آمیز/عادی) را انتخاب کن و همیشه دلیل دقیق بر اساس این اصول بنویس.

خروجی شما باید فقط به صورت یک شیء JSON با فیلدهای زیر باشد:
{"label": "", "hate_type": "", "reason": ""}

"""

def build_enhanced_prompt(text, examples):
    diverse_examples = []
    label_counts = Counter([ex['label'] for ex in examples])

    for label in ['تنفرآمیز', 'توهین‌آمیز', 'عادی']:
        label_examples = [ex for ex in examples if ex['label'] == label]
        diverse_examples.extend(label_examples[:2])

    ex_list = []
    for i, ex in enumerate(diverse_examples):
        ex_list.append(f"""مثال {i+1}:
متن: «{ex['text']}»
برچسب: {ex['label']}
نوع: {ex['hate_type']}
دلیل: {ex['reason']}""")

    ex_block = "\n\n".join(ex_list)

    prompt = f"""{PROMPT}

## نمونه‌های مرتبط:
{ex_block}

---
## متن تحلیل: «{text}»

تحلیل بر اساس اصول فوق:"""

    return prompt

def extract_json_enhanced(raw_response):
    try:
        cleaned_response = re.sub(r'``````', '', raw_response)

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)

            if all(k in parsed for k in ["label", "hate_type", "reason"]):
                if parsed["label"] not in ["تنفرآمیز", "توهین‌آمیز", "عادی"]:
                    parsed["label"] = "نامشخص"
                return parsed

        label_patterns = {
            r'تنفرآمیز|نفرت\s*آمیز|hate': 'تنفرآمیز',
            r'توهین\s*آمیز|توهین|insult': 'توهین‌آمیز',
            r'عادی|normal|neutral': 'عادی'
        }

        label = "نامشخص"
        for pattern, detected_label in label_patterns.items():
            if re.search(pattern, raw_response, re.IGNORECASE):
                label = detected_label
                break

        hate_type = "N/A"
        hate_priorities = [
            (r'قومی|ethnic', 'قومی'),
            (r'ملیتی|national|ایران', 'ملیتی'),
            (r'مذهبی|religious', 'مذهبی'),
            (r'جنسیتی|gender', 'جنسیتی'),
            (r'سیاسی|political', 'سیاسی'),
            (r'سنی|age|ageism', 'سنی')
        ]

        for pattern, detected_type in hate_priorities:
            if re.search(pattern, raw_response, re.IGNORECASE):
                hate_type = detected_type
                break

        return {
            "label": label,
            "hate_type": hate_type,
            "reason": "تحلیل خودکار از پاسخ مدل"
        }

    except Exception as e:
        return {
            "label": "نامشخص",
            "hate_type": "N/A",
            "reason": f"خطا در تجزیه: {str(e)}"
        }

# =============================================================================
# End of the COPIED block. Everything below is new.

REPO_COMMIT = "2bff8a083c05744ce2c9d92abcbfe26c10be88ce"
NOTEBOOK_RAW = f"https://raw.githubusercontent.com/hooman731/PHOND/{REPO_COMMIT}/Tag_Analysis.ipynb"

MODEL = "deepseek/deepseek-chat-v3-0324"
DECODING = {"max_tokens": 90, "temperature": 0.01, "top_p": 0.7}
TEXT_CHARS = 400
TOP_K = 6
MAX_TRIES = 3
API_RETRIES = 5
EMBEDDER = "xmanii/maux-gte-persian"
EMBEDDER_REVISION = "797be4fa81aa262c98516169d9c2254459a733e3"
EMBEDDER_CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"
EMBEDDER_LIBS = ["sentence-transformers==3.2.0", "transformers==4.45.2", "tokenizers==0.20.3", "huggingface_hub==0.25.2"]
EMBEDDER_ENV = Path.home() / ".cache" / "phond_encoder_env"
FALLBACK_REASON = "تحلیل خودکار از پاسخ مدل"
UNSPEC = "نامشخص"
FAILED = {"label": UNSPEC, "hate_type": "N/A", "reason": "عدم دریافت پاسخ معتبر پس از چندین تلاش"}

CONFIG = {
    "out": "phond_prompting_out",
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "provider": "siliconflow",
    "arms": "zero,fixed,dynamic",
    "fixed_sets": 3,
    "fixed_extra": "rare,common",
    "seed": 2026,
    "workers": 8,
    "n_boot": 2000,
    "price_in": 0.25,
    "price_out": 1.00,
}
CHARS_PER_TOKEN = 2.0
RETRY_FACTOR = 1.15

PAPER_COUNTS = {"normal": 25273, "offensive": 9026, "hateful": 1424}
HATE_TYPES = ["ethnic", "national", "gender", "age", "religious", "other", "political"]
LABELS3 = ["normal", "offensive", "hateful"]
LABELS9 = ["normal", "offensive"] + HATE_TYPES
ARM_ORDER = ["zero", "fixed", "fixed (mean)", "fixed_rare", "fixed_common", "dynamic"]

CORPUS = pd.DataFrame(examples)


_INVISIBLE = dict.fromkeys(map(ord, "‌‍‎‏‪‫‬‭‮﻿ـ"), None)
_DIACRITICS = re.compile(r"[ً-ٰٟ]")
_NUMERIC_LABELS = {"seen": False}


def norm_fa(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = s.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ة", "ه").replace("ۀ", "ه")
    s = s.translate(_INVISIBLE)
    s = _DIACRITICS.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def coarse_label(value):
    t = norm_fa(value).replace(" ", "")
    if not t:
        return None
    tl = t.lower()
    if "عادی" in t or tl in ("normal", "neutral"):
        return "normal"
    if "توهین" in t or tl in ("offensive", "offense", "insult"):
        return "offensive"
    if "تنفر" in t or "نفرت" in t or tl in ("hateful", "hate", "hatespeech"):
        return "hateful"
    if tl in ("0", "1", "2", "0.0", "1.0", "2.0"):
        _NUMERIC_LABELS["seen"] = True
        return LABELS3[int(float(tl))]
    return None


_NO_TYPE = {"n/a", "na", "none", "null", "nan", "-", "ندارد", "هیچ", "نامشخص", "بدون", "خیر"}
_TYPE_RULES = [
    ("ethnic", ("قوم", "نژاد"), r"ethnic|racial|\brace\b"),
    ("national", ("ملیت", "ملی"), r"national"),
    ("religious", ("مذهب", "دین"), r"religio"),
    ("gender", ("جنس",), r"gender|sexis|\bsex"),
    ("political", ("سیاس",), r"politic"),
    ("age", ("سنی", "مسن", "سالمند", "نسل"), r"\bage\b|ageis"),
    ("other", ("سایر", "دیگر"), r"\bother"),
]


def hate_type(value):
    raw = norm_fa(value)
    t = raw.replace(" ", "")
    if not t or t.lower() in _NO_TYPE:
        return None
    tl = raw.lower()
    for name, fa_keys, en_pat in _TYPE_RULES:
        if any(k in t for k in fa_keys) or re.search(en_pat, tl):
            return name
    if t == "سن":
        return "age"
    return None


def to3(label):
    return coarse_label(label) or "unspecified"


def to9(label, htype):
    c = to3(label)
    if c != "hateful":
        return c
    return hate_type(htype) or "other"


def unnamed_type(label, htype):
    return to3(label) == "hateful" and hate_type(htype) is None


def ref9(label, htype):
    c = coarse_label(label)
    if c != "hateful":
        return c
    return hate_type(htype) or "hateful_untyped"


def read_table(path: str, sheet=None) -> pd.DataFrame:
    if not Path(path).exists():
        sys.exit(f"File not found: {path}")
    if str(path).lower().endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path, dtype=str, sheet_name=sheet if sheet else 0).fillna("")
        except ValueError:
            sys.exit(f"Sheet '{sheet}' not found in {path}. Sheets: {pd.ExcelFile(path).sheet_names}")
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            sys.exit(f"The file is empty: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        print(f"  note: {Path(path).name} has invalid UTF-8 bytes, read with errors='replace'")
        return pd.read_csv(f, dtype=str, keep_default_na=False)


def pick_col(df, wanted, candidates, what, required=False):
    if wanted:
        if wanted not in df.columns:
            sys.exit(f"Column '{wanted}' ({what}) not found. Columns: {list(df.columns)}")
        return wanted
    low = {c.strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        sys.exit(f"Could not find the {what} column. Columns: {list(df.columns)}. Name it with the matching option.")
    return None


def norm_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Report:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self):
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def build_zero_prompt(text):
    return f"""{PROMPT}

---
## متن تحلیل: «{text}»

تحلیل بر اساس اصول فوق:"""


def corpus_rows(idx):
    return [{"text": CORPUS.at[i, "text"], "label": CORPUS.at[i, "label"],
             "hate_type": CORPUS.at[i, "hate_type"], "reason": CORPUS.at[i, "reason"]} for i in idx]


def make_fixed_sets(k: int, seed: int):
    lab = CORPUS["label"].map(coarse_label)
    hate_idx = CORPUS.index[lab == "hateful"].to_numpy()
    off_idx = CORPUS.index[lab == "offensive"].to_numpy()
    norm_idx = CORPUS.index[lab == "normal"].to_numpy()
    sets = []
    for s in range(k):
        rng = np.random.default_rng(seed + 100 + s)
        while True:
            h = rng.choice(hate_idx, 2, replace=False)
            if CORPUS.at[h[0], "hate_type"] != CORPUS.at[h[1], "hate_type"]:
                break
        o = rng.choice(off_idx, 2, replace=False)
        n = rng.choice(norm_idx, 2, replace=False)
        idx = [int(i) for i in list(h) + list(o) + list(n)]
        sets.append({"set": s + 1, "corpus_index": idx, "examples": corpus_rows(idx)})
    return sets


def make_type_sets(kinds, type_counts, base, seed):
    lab = CORPUS["label"].map(coarse_label)
    ctype = CORPUS["hate_type"].map(hate_type)
    types = [t for t in HATE_TYPES if ((lab == "hateful") & (ctype == t)).any()]
    order = sorted(types, key=lambda t: (type_counts.get(t, 0), HATE_TYPES.index(t)))
    if base is not None:
        rest = [int(i) for i in base["corpus_index"] if lab[i] != "hateful"]
    else:
        rng = np.random.default_rng(seed + 300)
        rest = [int(i) for i in list(rng.choice(CORPUS.index[lab == "offensive"].to_numpy(), 2, replace=False))
                + list(rng.choice(CORPUS.index[lab == "normal"].to_numpy(), 2, replace=False))]
    sets = []
    for kind in kinds:
        chosen = order[:2] if kind == "rare" else order[::-1][:2]
        rng = np.random.default_rng(seed + {"rare": 311, "common": 312}[kind])
        hate = [int(rng.choice(CORPUS.index[(lab == "hateful") & (ctype == t)].to_numpy())) for t in chosen]
        idx = hate + rest
        sets.append({"set": kind, "hate_types": chosen, "sample_counts": {t: int(type_counts.get(t, 0)) for t in types},
                     "corpus_index": idx, "examples": corpus_rows(idx)})
    return sets


def import_or_install(module, package):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        print(f"  installing {package} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
        try:
            return importlib.import_module(module)
        except ImportError:
            return None


ENCODER_JOB = r"""
import importlib.metadata
import json
import sys

HIDDEN = ("accelerate", "datasets", "peft")
for name in HIDDEN:
    sys.modules[name] = None
_metadata, _version = importlib.metadata.metadata, importlib.metadata.version


def _hidden_metadata(name):
    if name in HIDDEN:
        raise importlib.metadata.PackageNotFoundError(name)
    return _metadata(name)


def _hidden_version(name):
    if name in HIDDEN:
        raise importlib.metadata.PackageNotFoundError(name)
    return _version(name)


importlib.metadata.metadata, importlib.metadata.version = _hidden_metadata, _hidden_version

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

job = json.load(open(sys.argv[1], encoding="utf-8"))
pin = {"code_revision": job["code_revision"]} if job["code_revision"] else {}
model = SentenceTransformer(job["model"], trust_remote_code=True, revision=job["revision"],
                            config_kwargs=pin, model_kwargs=pin)
corpus = np.asarray(model.encode(job["corpus"], show_progress_bar=False), dtype=np.float32)
queries = np.stack([np.asarray(model.encode([t], normalize_embeddings=True), dtype=np.float32)[0]
                    for t in job["queries"]])
hf = model[0].auto_model
info = {"sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
        "torch": torch.__version__, "device": str(model.device), "dtype": str(hf.dtype),
        "model_commit": getattr(hf.config, "_commit_hash", None), "code_module": type(hf).__module__}
np.save(job["corpus_out"], corpus)
np.save(job["queries_out"], queries)
with open(job["info_out"], "w", encoding="utf-8") as f:
    json.dump(info, f, indent=1)
"""


def encoder_environment(log):
    stamp = EMBEDDER_ENV / "installed.txt"
    want = " ".join(EMBEDDER_LIBS)
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == want:
        return EMBEDDER_ENV
    log(f"  installing {want} in {EMBEDDER_ENV}, for the encoder only ...")
    if EMBEDDER_ENV.exists():
        shutil.rmtree(EMBEDDER_ENV)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                        "--target", str(EMBEDDER_ENV), *EMBEDDER_LIBS], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("pip could not install the encoder's library versions:\n" + (r.stderr or r.stdout)[-1500:])
    stamp.write_text(want, encoding="utf-8")
    return EMBEDDER_ENV


def encode_like_annotation(queries, log):
    env_dir = encoder_environment(log)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        job = {"model": EMBEDDER, "revision": EMBEDDER_REVISION, "code_revision": EMBEDDER_CODE_REVISION,
               "corpus": CORPUS["text"].tolist(), "queries": list(queries),
               "corpus_out": str(tmp / "corpus.npy"), "queries_out": str(tmp / "queries.npy"),
               "info_out": str(tmp / "info.json")}
        (tmp / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        (tmp / "encode.py").write_text(ENCODER_JOB, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(env_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env.update({"USE_TF": "0", "USE_FLAX": "0", "USE_TORCH": "1", "TOKENIZERS_PARALLELISM": "false"})
        r = subprocess.run([sys.executable, str(tmp / "encode.py"), str(tmp / "job.json")], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("the encoder run failed:\n" + "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:]))
        corpus = np.load(tmp / "corpus.npy")
        qv = np.load(tmp / "queries.npy")
        info = json.loads((tmp / "info.json").read_text(encoding="utf-8"))
    want = dict(s.split("==") for s in EMBEDDER_LIBS)
    if (info["sentence_transformers"], info["transformers"]) != (want["sentence-transformers"], want["transformers"]):
        raise RuntimeError(f"the encoder ran with sentence-transformers {info['sentence_transformers']} and "
                           f"transformers {info['transformers']}, not with the versions of its files")
    if EMBEDDER_REVISION and info.get("model_commit") != EMBEDDER_REVISION:
        raise RuntimeError(f"the encoder loaded commit {info.get('model_commit')}, not {EMBEDDER_REVISION}")
    if EMBEDDER_CODE_REVISION and EMBEDDER_CODE_REVISION not in str(info.get("code_module")):
        raise RuntimeError(f"the encoder's model code came from {info.get('code_module')}, not from commit "
                           f"{EMBEDDER_CODE_REVISION}")
    if not (np.isfinite(corpus).all() and np.isfinite(qv).all()):
        raise RuntimeError("the encoder gave vectors with NaN or inf")
    notes = [f"{EMBEDDER} at commit {str(info.get('model_commit'))[:7]}, model code {info.get('code_module')}",
             f"run with sentence-transformers {info['sentence_transformers']} and transformers {info['transformers']} "
             f"(the versions recorded in the encoder's files); torch {info['torch']}, {info['device']}, {info['dtype']}"]
    return corpus, qv, notes


class Retriever:
    def __init__(self, texts, log):
        emb, self.queries, self.notes = encode_like_annotation(texts, log)
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        faiss = import_or_install("faiss", "faiss-cpu")
        if faiss is not None:
            faiss.normalize_L2(emb)
            self.index = faiss.IndexHNSWFlat(emb.shape[1], 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
            self.index.add(emb)
            self.kind = "faiss HNSW, as in the notebook"
        else:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            self.emb, self.index, self.kind = emb, None, "exact cosine search (faiss could not be installed)"

    def similar(self, i, top_k=TOP_K):
        q = np.ascontiguousarray(self.queries[i:i + 1], dtype=np.float32)
        if self.index is not None:
            _, idx = self.index.search(q, top_k)
            idx = [int(j) for j in idx[0] if 0 <= j < len(CORPUS)]
        else:
            idx = [int(j) for j in np.argsort(-(self.emb @ q[0]), kind="stable")[:top_k]]
        return corpus_rows(idx)

class FatalAPIError(RuntimeError):
    pass


class TransientAPIError(RuntimeError):
    pass


def call_api(prompt, cfg):
    data = {"model": MODEL,
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": prompt}],
            **DECODING}
    if cfg["provider"]:
        data["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": False}
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    last = "no answer"
    for attempt in range(API_RETRIES):
        t0 = time.time()
        try:
            r = requests.post(cfg["api_url"], headers=headers, json=data, timeout=80)
        except requests.RequestException as e:
            last = f"network error: {e}"[:300]
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429:
            last = "HTTP 429 (rate limit)"
            time.sleep(min(5 * (2 ** attempt), 60))
            continue
        if r.status_code == 403 and "moderat" in r.text.lower():
            return "", {"provider": None, "tokens_in": 0, "tokens_out": 0, "cost": None,
                        "seconds": round(time.time() - t0, 2), "note": "moderation"}
        if r.status_code in (400, 401, 402, 403, 404):
            raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:400]}")
        if r.status_code != 200:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(3 * (attempt + 1))
            continue
        try:
            j = r.json()
            content = (j["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            last = f"unexpected answer: {r.text[:200]}"
            time.sleep(3 * (attempt + 1))
            continue
        usage = j.get("usage") or {}
        return content, {"provider": j.get("provider"), "tokens_in": usage.get("prompt_tokens") or 0,
                         "tokens_out": usage.get("completion_tokens") or 0, "cost": usage.get("cost"),
                         "seconds": round(time.time() - t0, 2)}
    raise TransientAPIError(last)


def classify(prompt, cfg):
    tries = []
    for _ in range(MAX_TRIES):
        raw, meta = call_api(prompt, cfg)
        tries.append({"raw": raw, **meta})
        if raw:
            result = extract_json_enhanced(raw)
            try:
                if (result["label"] != UNSPEC and
                        len(result.get("reason", "")) > 8 and
                        result["reason"] != FALLBACK_REASON):
                    return result, tries
            except Exception:
                pass
    return dict(FAILED), tries


def confusion(yk, pk, n_lab, n_cat, w):
    return np.bincount(yk * n_cat + pk, weights=w, minlength=n_lab * n_cat).reshape(n_lab, n_cat)


def scores_batch(Ms, n_lab):
    Ms = np.asarray(Ms, dtype=float)
    total = Ms.sum(axis=(1, 2))
    d = np.arange(n_lab)
    tp = Ms[:, d, d]
    support = Ms.sum(axis=2)
    predicted = Ms[:, :, :n_lab].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        present = support > 0
        npres = present.sum(axis=1)

        def macro(v):
            return np.where(npres > 0, (v * present).sum(axis=1) / np.maximum(npres, 1), np.nan)

        acc = np.where(total > 0, tp.sum(axis=1) / total, np.nan)
        pe = np.where(total > 0, (support * predicted).sum(axis=1) / total ** 2, np.nan)
        kappa = np.where(pe < 1, (acc - pe) / (1 - pe), np.nan)
    return {"accuracy": acc, "macro_P": macro(prec), "macro_R": macro(rec), "macro_F1": macro(f1),
            "kappa": kappa, "P": prec, "R": rec, "F1": f1, "support": support, "predicted": predicted}


def boot_scores(yk, pk, wk, n_lab, n_cat, boot):
    B = boot.shape[0]
    flat = yk[boot] * n_cat + pk[boot] + (np.arange(B) * (n_lab * n_cat))[:, None]
    Ms = np.bincount(flat.ravel(), weights=wk[boot].ravel(), minlength=B * n_lab * n_cat)
    return scores_batch(Ms.reshape(B, n_lab, n_cat), n_lab)


def mcnemar_p(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def stage2_scores(y3, y9, p3, p9, n_boot=0, seed=0, unnamed=None):
    H = np.flatnonzero([(a == "hateful") and (b in HATE_TYPES) for a, b in zip(y3, y9)])
    if len(H) == 0:
        return None
    L = len(HATE_TYPES)
    yk = np.array([HATE_TYPES.index(v) for v in y9[H]], dtype=int)
    both = np.array([v == "hateful" for v in p3[H]])
    pk = np.array([HATE_TYPES.index(v) if v in HATE_TYPES else L for v in p9[H]], dtype=int)
    wk = both.astype(float)
    s = scores_batch(confusion(yk, pk, L, L + 1, wk)[None], L)
    as_other = int((np.asarray(unnamed, dtype=bool)[H] & both).sum()) if unnamed is not None else 0
    res = {"ref_hateful": len(H), "both_hateful": int(both.sum()), "unnamed_as_other": as_other,
           "agreement": float(s["accuracy"][0]), "kappa": float(s["kappa"][0]), "macro_F1": float(s["macro_F1"][0])}
    bs = None
    if n_boot:
        boot = np.random.default_rng(seed).integers(0, len(H), size=(n_boot, len(H)))
        bs = boot_scores(yk, pk, wk, L, L + 1, boot)
    return res, bs


def label_index(df, lab_col, type_col, text_col, id_col):
    by_id, by_text, ambiguous = {}, {}, set()
    for k in range(len(df)):
        rec = (df[lab_col].iat[k], df[type_col].iat[k] if type_col else "N/A")
        nt = norm_fa(df[text_col].iat[k]) if text_col else None
        if id_col and str(df[id_col].iat[k]).strip():
            by_id.setdefault(norm_id(df[id_col].iat[k]), (rec, nt))
        if nt:
            if nt in by_text and by_text[nt] != rec:
                ambiguous.add(nt)
            by_text.setdefault(nt, rec)
    return {"by_id": by_id, "by_text": by_text, "ambiguous": ambiguous, "rows": len(df)}


def released_from_dataset(path, ev, text_col, id_col, log):
    d = read_table(path)
    d_lab = pick_col(d, None, ["classification"], "label")
    d_text = pick_col(d, None, ["Text", "text"], "text")
    if not (d_lab and d_text):
        log(f"  note: {Path(path).name} has no classification or Text column, so the released labels are left out")
        return None, None
    index = label_index(d, d_lab, pick_col(d, None, ["hate_class"], "hate type"), d_text,
                        pick_col(d, None, ["Id", "id", "ID"], "id"))
    recs, st = match_labels(ev[text_col], ev[id_col] if id_col else [None] * len(ev), index)
    log(f"  the sample has no released label columns; taken from {Path(path).name}: matched "
        + ", ".join(f"{k} {v}" for k, v in st.items()))
    if st["not found"]:
        log(f"  WARNING: {st['not found']} texts of the sample are not in the dataset file, so the released labels "
            "are left out. Add the released label columns to the sample, or name them with --rel_col and --rel_type_col.")
        return None, None
    ev["released_label_from_dataset"] = [r[0] for r in recs]
    ev["released_type_from_dataset"] = [r[1] for r in recs]
    return "released_label_from_dataset", "released_type_from_dataset"


def match_labels(texts, ids, index):
    out, stats = [], Counter()
    for t, i in zip(texts, ids):
        nt = norm_fa(t)
        rec = None
        if i is not None and str(i).strip() and index["by_id"]:
            hit = index["by_id"].get(norm_id(i))
            if hit is not None:
                rec_id, txt_id = hit
                if txt_id is None or txt_id == nt or txt_id[:60] == nt[:60]:
                    rec = rec_id
                    stats["by Id"] += 1
                else:
                    stats["Id found but text differs"] += 1
        if rec is None and nt in index["by_text"]:
            rec = index["by_text"][nt]
            stats["by text"] += 1
            if nt in index["ambiguous"]:
                stats["by a text that repeats with different labels (first one used)"] += 1
        if rec is None:
            stats["not found"] += 1
        out.append(rec)
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="evaluation sample (csv or xlsx)")
    ap.add_argument("--sheet", default=None, help="sheet name, when --eval is an Excel file (default: the first sheet)")
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--rel_col", default=None, help="released label column")
    ap.add_argument("--rel_type_col", default=None, help="released hate type column")
    ap.add_argument("--weight_col", default=None)
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--dataset", default=None, help="PHOND CSV, to take the released labels from when the sample has none")
    ap.add_argument("--arms", default=CONFIG["arms"], help="any of zero,fixed,dynamic (default: all three)")
    ap.add_argument("--fixed_sets", type=int, default=CONFIG["fixed_sets"], help="number of fixed sets drawn at random")
    ap.add_argument("--fixed_extra", default=CONFIG["fixed_extra"],
                    help="fixed sets chosen by the hate types of the sample: rare (the two rarest), common (the two "
                         "most common), both (default) or none (empty)")
    ap.add_argument("--provider", default=CONFIG["provider"], help="OpenRouter provider slug, or none")
    ap.add_argument("--api_url", default=CONFIG["api_url"])
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--workers", type=int, default=CONFIG["workers"])
    ap.add_argument("--n_boot", type=int, default=CONFIG["n_boot"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N texts, for a cheap test")
    ap.add_argument("--out", default=CONFIG["out"])
    ap.add_argument("--dry_run", action="store_true", help="show columns, prompts and cost, call nothing")
    ap.add_argument("--verify_repo", action="store_true", help="compare the copied prompt and corpus with GitHub")
    ap.add_argument("--notebook", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Report(out / "report.txt")
    try:
        run(args, out, log)
    finally:
        log.save()


def run(args, out, log):
    provider = None if str(args.provider).strip().lower() in ("", "none", "any", "auto") else args.provider.strip().lower()
    arms_req = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms_req) - {"zero", "fixed", "dynamic"}
    if unknown:
        sys.exit(f"Unknown arms {sorted(unknown)}. Use any of zero,fixed,dynamic.")
    log("PHOND: effect of the examples in the labeling prompt")
    log(f"run at {time.strftime('%Y-%m-%d %H:%M')}; model {MODEL}; {DECODING}; provider {provider or 'chosen by OpenRouter'}")
    log(f"arms {arms_req}; fixed sets {args.fixed_sets}; seed {args.seed}; bootstrap {args.n_boot}")
    log(f"prompt and corpus from Tag_Analysis.ipynb, commit {REPO_COMMIT[:7]}: {len(CORPUS)} examples "
        f"{dict(Counter(CORPUS['label'].map(coarse_label)))}")
    if args.verify_repo or args.notebook:
        verify_repo(log, args.notebook)

    ev = read_table(args.eval, args.sheet)
    ref_name = "released labels"
    text_col = pick_col(ev, args.text_col, ["text", "متن"], "text", required=True)
    weight_col = pick_col(ev, args.weight_col, ["weight_to_full", "weight"], "weight")
    id_col = pick_col(ev, args.id_col, ["Id", "id"], "id")
    ref_col = pick_col(ev, args.rel_col, ["released_label", "classification", "label_a", "label1", "A"], "released label")
    ref_type_col = pick_col(ev, args.rel_type_col, ["released_hate_type", "hate_class", "type_a", "hate_type_a"],
                            "released hate type")
    if not ref_col and args.dataset:
        ref_col, ref_type_col = released_from_dataset(args.dataset, ev, text_col, id_col, log)
    if not ref_col:
        sys.exit(f"Could not find the released label column. Columns: {list(ev.columns)}. "
                 "Name it with --rel_col and --rel_type_col, or give the PHOND CSV with --dataset.")
    log(f"\nsample {args.eval}: {len(ev)} rows")
    log(f"  columns used: text={text_col} | reference = released labels: {ref_col}, type={ref_type_col} | "
        f"weight={weight_col} | id={id_col}")
    log("  CHECK these columns. If one is wrong, name it with --rel_col and --rel_type_col.")

    ev = ev.reset_index(drop=True)
    no_types = [None] * len(ev)
    ev["_ref3"] = ev[ref_col].map(coarse_label)
    ev["_ref9"] = [ref9(a, b) for a, b in zip(ev[ref_col], ev[ref_type_col] if ref_type_col else no_types)]
    bad = ev["_ref3"].isna() | (ev[text_col].map(norm_fa) == "")
    if bad.any():
        log(f"  {int(bad.sum())} rows have no usable reference label ({ref_name}) or no text and are left out. "
            f"Label values there: {sorted(ev.loc[bad, ref_col].astype(str).unique())[:8]}")
    ev = ev.loc[~bad].reset_index(drop=True)
    if args.limit:
        ev = ev.head(args.limit).copy()
    if _NUMERIC_LABELS["seen"]:
        log("  WARNING: numeric labels were read as 0 normal, 1 offensive, 2 hateful. Check that this is right.")
    n = len(ev)
    if n == 0:
        sys.exit("No usable rows in the sample.")
    texts = [str(t)[:TEXT_CHARS] for t in ev[text_col]]
    log(f"  texts evaluated: {n}")
    log(f"  {ref_name}: {dict(Counter(ev['_ref3']))}"
        + (f"; hate types {dict(Counter(v for v in ev['_ref9'] if v not in ('normal', 'offensive')))}"
           if ref_type_col else ""))
    if not ref_type_col:
        log(f"  note: no hate type column for the {ref_name}, so the nine class results are left out")
    show = [c for c in [text_col, ref_col, ref_type_col] if c]
    log("  first rows:")
    for _, r in ev[show].head(3).iterrows():
        log("    " + " | ".join(f"{c}: {str(r[c])[:40]}" for c in show))

    w = None
    if weight_col:
        w = pd.to_numeric(ev[weight_col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(w).any() or (w <= 0).any():
            log("  note: some weights are missing, so the reweighted results are left out")
            w = None
        else:
            log(f"  weights: from the column {weight_col}")
    elif ev["_ref3"].notna().all():
        cls3 = "_ref3"
        tot = sum(PAPER_COUNTS.values())
        share = ev[cls3].value_counts() / n
        w = ev[cls3].map(lambda c: (PAPER_COUNTS[c] / tot) / share[c]).to_numpy(dtype=float)
        log("  weights: share of each released class in the full dataset over its share in the sample: "
            + ", ".join(f"{c} {(PAPER_COUNTS[c] / tot) / share[c]:.4f}" for c in LABELS3 if c in share))

    plan = []
    if "zero" in arms_req:
        plan.append(("zero", [build_zero_prompt(t) for t in texts]))
    fixed_sets = make_fixed_sets(args.fixed_sets, args.seed) if "fixed" in arms_req else []
    kinds = [k.strip().lower() for k in str(args.fixed_extra or "").split(",") if k.strip()]
    kinds = ["rare", "common"] if kinds == ["both"] else ([] if kinds == ["none"] else kinds)
    if set(kinds) - {"rare", "common"}:
        sys.exit(f"Unknown --fixed_extra {sorted(set(kinds) - {'rare', 'common'})}. Use rare, common or both.")
    if "fixed" in arms_req and kinds:
        sample_types = Counter(v for v in ev["_ref9"] if v in HATE_TYPES)
        fixed_sets += make_type_sets(kinds, sample_types, fixed_sets[0] if fixed_sets else None, args.seed)
    (out / "fixed_sets.json").write_text(json.dumps(fixed_sets, ensure_ascii=False, indent=1), encoding="utf-8")
    for fs in fixed_sets:
        name = f"fixed{fs['set']}" if isinstance(fs["set"], int) else f"fixed_{fs['set']}"
        plan.append((name, [build_enhanced_prompt(t, fs["examples"]) for t in texts]))
    if fixed_sets:
        log(f"\nfixed examples (corpus rows), saved in {out / 'fixed_sets.json'}:")
        for fs in fixed_sets:
            log(f"  set {fs['set']}: " + "; ".join(f"{e['label']}/{e['hate_type']}: {e['text']}" for e in fs["examples"]))
        typed = [fs for fs in fixed_sets if not isinstance(fs["set"], int)]
        if typed:
            log(f"  hate types in the sample: {typed[0]['sample_counts']}")
            for fs in typed:
                log(f"  set {fs['set']}: the two hate types {'rarest' if fs['set'] == 'rare' else 'most common'} in the "
                    f"sample ({', '.join(fs['hate_types'])}); offensive and normal examples as in set 1")
    if "dynamic" in arms_req:
        log(f"\ndynamic arm: retrieval with the encoder of the annotation, {EMBEDDER}")
        try:
            retr = Retriever(texts, log)
            for note in retr.notes:
                log(f"  encoder: {note}")
            log(f"  search: {retr.kind}")
            retrieved = [retr.similar(i) for i in range(len(texts))]
            plan.append(("dynamic", [build_enhanced_prompt(t, ex) for t, ex in zip(texts, retrieved)]))
            retrieval_stats(retrieved, ev["_ref3"].tolist(), log, out)
        except Exception as e:
            import traceback
            tail = traceback.format_exc().strip().splitlines()[-8:]
            log(f"\nWARNING: the dynamic arm is skipped, the retriever could not be loaded: "
                f"{type(e).__name__}: {str(e)[:300]}")
            log("  end of the error:\n    " + "\n    ".join(tail))
            log("  The other arms are not affected.")

    with open(out / "prompt_preview.txt", "w", encoding="utf-8") as f:
        f.write(f"SYSTEM MESSAGE (all arms)\n{PROMPT}\n\n")
        for name, prompts in plan:
            f.write(f"===== {name}: USER MESSAGE for text 1 =====\n{prompts[0]}\n\n")
    if plan:
        calls = sum(len(set(p)) for _, p in plan) * RETRY_FACTOR
        chars = np.mean([len(PROMPT) + len(x) for _, p in plan for x in p])
        cost = calls * (chars / CHARS_PER_TOKEN * CONFIG["price_in"] + DECODING["max_tokens"] * CONFIG["price_out"]) / 1e6
        log(f"\nplanned: {len(plan)} arms x {n} texts; about {int(calls)} calls with retries; "
            f"about ${cost:.2f} (rough guess: {CHARS_PER_TOKEN:g} characters per token, prices of {CONFIG['price_in']} "
            f"and {CONFIG['price_out']} USD per million tokens)")
        log(f"  the system message and the first user message of each arm are in {out / 'prompt_preview.txt'}")

    if args.dry_run:
        log("\ndry run: nothing was sent to the API")
        return

    preds, missing = {}, {}
    if plan:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("OPENROUTER_API_KEY")
            except Exception:
                api_key = None
        if not api_key:
            sys.exit("No API key. In a Colab cell run: import os; from google.colab import userdata; "
                     "os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')")
        cfg = {"api_key": api_key, "api_url": args.api_url, "provider": provider}
        preds, missing = run_arms(plan, cfg, out, args.workers, log)

    if not preds:
        sys.exit("Nothing to evaluate: no arm ran.")

    evaluate(ev, preds, w, args, out, log, has_types=ref_type_col is not None, text_col=text_col)

    if any(missing.values()):
        log(f"\nINCOMPLETE: {sum(missing.values())} answers are missing (API errors) and were scored as unspecified. "
            "Run the same command again; only the missing texts are sent, and all numbers are computed again.")
    log(f"\nreport written to {out / 'report.txt'}")


def run_arms(plan, cfg, out, workers, log):
    cache_path = out / "cache.jsonl"
    pin = cfg["provider"] or "auto"
    done, other = {}, 0
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("pin") != pin:
                other += 1
                continue
            done[(rec["arm"], rec["ph"])] = rec
    tasks, seen = [], set()
    for name, prompts in plan:
        for p in prompts:
            k = (name, sha16(p))
            if k not in done and k not in seen:
                seen.add(k)
                tasks.append((name, k[1], p))
    log(f"\nlabeling: {len(tasks)} prompts to send; {len(done)} answers already in the cache"
        + (f"; {other} cached answers from another provider setting are not used" if other else ""))

    lock = threading.Lock()
    stop = threading.Event()
    state = {"in_a_row": 0, "fatal": None, "finished": 0}
    limit_in_a_row = max(24, 3 * workers)
    start = time.time()

    def work(task):
        name, ph, prompt = task
        if stop.is_set():
            return
        try:
            result, tries = classify(prompt, cfg)
        except FatalAPIError as e:
            with lock:
                state["fatal"] = state["fatal"] or str(e)
            stop.set()
            return
        except TransientAPIError as e:
            with lock:
                state["in_a_row"] += 1
                if state["in_a_row"] >= limit_in_a_row:
                    state["fatal"] = state["fatal"] or f"{state['in_a_row']} texts in a row got no answer. Last error: {e}"
                    stop.set()
            return
        rec = {"arm": name, "ph": ph, "pin": pin, "result": result, "tries": tries}
        with lock:
            state["in_a_row"] = 0
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(name, ph)] = rec
            state["finished"] += 1
            k = state["finished"]
        if k % 50 == 0 or k == len(tasks):
            print(f"  {k}/{len(tasks)} done, {time.time() - start:.0f}s", flush=True)

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for fut in as_completed([pool.submit(work, t) for t in tasks]):
                fut.result()
    if state["fatal"]:
        sys.exit(f"\nThe run stopped. {state['fatal']}\n"
                 "If the provider is the problem, run again with --provider none (or another provider, "
                 "such as --provider gmicloud); this labels every text again with that setting. "
                 "Answers received so far are kept in the cache.")

    preds, run_info, missing = {}, {}, {}
    for name, prompts in plan:
        recs = [done.get((name, sha16(p))) for p in prompts]
        missing[name] = sum(r is None for r in recs)
        preds[name] = [r["result"] if r else dict(FAILED) for r in recs]
        uniq = {sha16(p): done.get((name, sha16(p))) for p in prompts}
        tries = [t for r in uniq.values() if r for t in r["tries"]]
        tin = sum(t.get("tokens_in") or 0 for t in tries)
        tout = sum(t.get("tokens_out") or 0 for t in tries)
        reported = [t.get("cost") for t in tries if t.get("cost") is not None]
        cost = sum(reported) if tries and len(reported) == len(tries) else \
            (tin * CONFIG["price_in"] + tout * CONFIG["price_out"]) / 1e6
        answered = sum(1 for r in uniq.values() if r)
        run_info[name] = {"texts": len(prompts), "unique prompts": len(uniq), "answered": answered,
                          "calls": len(tries), "tokens in": tin, "tokens out": tout, "cost USD": round(cost, 4),
                          "seconds per text": round(sum(t.get("seconds") or 0 for t in tries) / max(answered, 1), 2),
                          "providers": dict(Counter(str(t.get("provider")) for t in tries))}
    log("\n=== Tokens, cost and time per arm ===")
    for name, ri in run_info.items():
        log(f"  {name:8s} texts {ri['texts']}, calls {ri['calls']} ({ri['calls'] / max(ri['answered'], 1):.2f} per text), "
            f"tokens in {ri['tokens in']:,} out {ri['tokens out']:,}, cost ${ri['cost USD']:.3f}, "
            f"{ri['seconds per text']:.2f}s of API time per text, providers {ri['providers']}")
    if run_info:
        log(f"  total ${sum(r['cost USD'] for r in run_info.values()):.2f} (OpenRouter's own cost when it reports one, "
            "otherwise tokens times the prices in CONFIG)")
    pd.DataFrame(run_info).T.to_csv(out / "cost.csv", encoding="utf-8-sig")
    if any(missing.values()):
        log(f"  WARNING: {sum(missing.values())} answers are missing because the API gave no answer "
            f"({ {k: v for k, v in missing.items() if v} }). They are scored as unspecified below. "
            "Run the same command again to fill them in.")
    return preds, missing


def retrieval_stats(retrieved, ref3, log, out):
    rows = []
    for i, exs in enumerate(retrieved):
        c = Counter(coarse_label(e["label"]) for e in exs)
        rows.append({"text_no": i + 1, "reference": ref3[i],
                     "hateful_retrieved": c.get("hateful", 0), "offensive_retrieved": c.get("offensive", 0),
                     "normal_retrieved": c.get("normal", 0),
                     "hateful_in_prompt": min(c.get("hateful", 0), 2),
                     "examples_in_prompt": sum(min(c.get(k, 0), 2) for k in LABELS3)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "retrieval.csv", index=False, encoding="utf-8-sig")
    log("\n=== Dynamic examples (retrieval step of the pipeline) ===")
    log(f"  of the {TOP_K} nearest corpus examples, on average {df.hateful_retrieved.mean():.2f} are hateful, "
        f"{df.offensive_retrieved.mean():.2f} offensive and {df.normal_retrieved.mean():.2f} normal")
    dist = Counter(df.examples_in_prompt)
    log(f"  examples that enter the prompt (at most 2 per class): mean {df.examples_in_prompt.mean():.2f}; "
        + ", ".join(f"{k} examples: {dist[k]} texts" for k in sorted(dist)))
    log(f"  texts with a hateful example in the prompt: {(df.hateful_in_prompt > 0).mean():.1%}; "
        f"with 4 or more hateful among the {TOP_K} retrieved: {(df.hateful_retrieved >= 4).mean():.1%}")
    for c in LABELS3:
        g = df[df.reference == c]
        if len(g):
            log(f"  texts with reference label {c} ({len(g)}): {g.hateful_retrieved.mean():.2f} hateful of {TOP_K} "
                f"retrieved, {(g.hateful_in_prompt > 0).mean():.1%} with a hateful example in the prompt")


def evaluate(ev, preds, w, args, out, log, has_types, text_col):
    main_ref = "released labels"
    main_short = "released"
    n = len(ev)
    arms = list(preds)
    fixed = [a for a in arms if re.fullmatch(r"fixed\d+", a)]
    p3 = {a: np.array([to3(r["label"]) for r in preds[a]], dtype=object) for a in arms}
    p9 = {a: np.array([to9(r["label"], r.get("hate_type")) for r in preds[a]], dtype=object) for a in arms}
    refs = {main_ref: (ev["_ref3"].to_numpy(dtype=object), ev["_ref9"].to_numpy(dtype=object))}
    levels = [("3", LABELS3)] + ([("9", LABELS9)] if has_types else [])
    weightings = [("raw", np.ones(n))] + ([("reweighted", np.asarray(w, dtype=float))] if w is not None else [])
    conf_dir = out / "confusion"
    conf_dir.mkdir(exist_ok=True)

    unnamed = {a: np.array([unnamed_type(r["label"], r.get("hate_type")) for r in preds[a]]) for a in arms}
    log("")
    for a in arms:
        names = Counter(str(r.get("hate_type")) for r, u in zip(preds[a], unnamed[a]) if u)
        if names:
            log(f"  note: {a}: {sum(names.values())} hateful labels name none of the six defined hate types and "
                f"count as other: {dict(names.most_common(6))}")

    rows, per_class, boots, correct = [], [], {}, {}
    for ref_name, (y3, y9) in refs.items():
        for lev, labels in levels:
            y = y3 if lev == "3" else y9
            idx = np.flatnonzero([v in labels for v in y])
            if len(idx) == 0:
                continue
            if lev == "9" and len(idx) < n:
                log(f"  note: {n - len(idx)} texts have no usable hate type in the {ref_name} labels and are left out "
                    "of the nine class results against them")
            L = len(labels)
            ypos = {c: i for i, c in enumerate(labels)}
            yk = np.array([ypos[v] for v in y[idx]])
            boot = np.random.default_rng(args.seed + 7).integers(0, len(idx), size=(args.n_boot, len(idx)))
            for a in arms:
                p = (p3 if lev == "3" else p9)[a][idx]
                cats = list(labels) + sorted(set(p) - set(labels))
                cpos = {c: i for i, c in enumerate(cats)}
                pk = np.array([cpos[v] for v in p])
                C = len(cats)
                if ref_name == main_ref:
                    correct[(a, lev)] = yk == pk
                for wname, wv in weightings:
                    wk = wv[idx]
                    M = confusion(yk, pk, L, C, wk)
                    s = scores_batch(M[None], L)
                    bs = boot_scores(yk, pk, wk, L, C, boot)
                    boots[(ref_name, a, lev, wname)] = bs
                    row = {"arm": a, "reference": ref_name, "level": lev, "weighting": wname, "n": len(idx)}
                    for m in ("accuracy", "macro_P", "macro_R", "macro_F1", "kappa"):
                        row[m] = float(s[m][0])
                    for m in ("accuracy", "macro_F1", "kappa"):
                        row[f"{m}_lo"], row[f"{m}_hi"] = ci(bs[m])
                    row["pred_hateful_share"] = float(np.average(p3[a][idx] == "hateful", weights=wk))
                    row["ref_hateful_share"] = float(np.average(y3[idx] == "hateful", weights=wk))
                    row["unspecified"] = int((p == "unspecified").sum())
                    rows.append(row)
                    if wname == "raw":
                        for j, c in enumerate(labels):
                            per_class.append({"arm": a, "reference": ref_name, "level": lev, "class": c,
                                              "support": int(s["support"][0][j]),
                                              "predicted": int(s["predicted"][0][j]),
                                              "P": round(float(s["P"][0][j]), 4), "R": round(float(s["R"][0][j]), 4),
                                              "F1": round(float(s["F1"][0][j]), 4)})
                        short = main_short
                        pd.DataFrame(M.astype(int), index=[f"{ref_name}: {c}" for c in labels],
                                     columns=[f"{a}: {c}" for c in cats]) \
                            .to_csv(conf_dir / f"{a}_{lev}class_vs_{short}.csv", encoding="utf-8-sig")

    met = pd.DataFrame(rows)
    if len(fixed) > 1:
        extra = []
        for (ref_name, lev, wname), g in met[met.arm.isin(fixed)].groupby(["reference", "level", "weighting"], sort=False):
            row = {"arm": "fixed (mean)", "reference": ref_name, "level": lev, "weighting": wname, "n": int(g.n.iloc[0])}
            for m in ("accuracy", "macro_P", "macro_R", "macro_F1", "kappa", "pred_hateful_share", "ref_hateful_share"):
                row[m] = float(g[m].mean())
            bs = {m: np.mean([boots[(ref_name, a, lev, wname)][m] for a in fixed], axis=0)
                  for m in ("accuracy", "macro_F1", "kappa")}
            boots[(ref_name, "fixed (mean)", lev, wname)] = bs
            for m in ("accuracy", "macro_F1", "kappa"):
                row[f"{m}_lo"], row[f"{m}_hi"] = ci(bs[m])
            row["unspecified"] = int(round(g.unspecified.mean()))
            row["sd_over_sets_macro_F1"] = float(g.macro_F1.std())
            row["sd_over_sets_kappa"] = float(g.kappa.std())
            extra.append(row)
        met = pd.concat([met, pd.DataFrame(extra)], ignore_index=True)

    def arm_rank(a):
        base = re.sub(r"\d+$", "", a)
        return ARM_ORDER.index(base) if base in ARM_ORDER else 99

    met["_o"] = met.arm.map(arm_rank)
    met["_r"] = (met.reference != main_ref).astype(int)
    met = met.sort_values(["_r", "level", "weighting", "_o", "arm"], kind="stable").drop(columns=["_o", "_r"])
    met.round(4).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False, encoding="utf-8-sig")

    def get(a, lev, wname, ref_name=main_ref):
        g = met[(met.arm == a) & (met.reference == ref_name) & (met.level == lev) & (met.weighting == wname)]
        return g.iloc[0] if len(g) else None

    fx = "fixed (mean)" if len(fixed) > 1 else (fixed[0] if fixed else None)
    pairs = [("dynamic", "zero"), ("dynamic", fx), (fx, "zero"), ("dynamic", "fixed_rare"), ("dynamic", "fixed_common")]
    comps = []
    for a, b in pairs:
        if not a or not b:
            continue
        for lev, _ in levels:
            for wname, _ in weightings:
                ka, kb = (main_ref, a, lev, wname), (main_ref, b, lev, wname)
                if ka not in boots or kb not in boots:
                    continue
                ra, rb = get(a, lev, wname), get(b, lev, wname)
                for m in ("macro_F1", "kappa", "accuracy"):
                    dlt = boots[ka][m] - boots[kb][m]
                    dlt = dlt[~np.isnan(dlt)]
                    p_boot = min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean())) if len(dlt) else np.nan
                    mc = np.nan
                    if m == "accuracy" and wname == "raw" and (a, lev) in correct and (b, lev) in correct:
                        mc = mcnemar_p(correct[(a, lev)], correct[(b, lev)])
                    lo, hi = ci(dlt)
                    comps.append({"a": a, "b": b, "level": lev, "weighting": wname, "metric": m,
                                  "a_value": ra[m], "b_value": rb[m], "diff": ra[m] - rb[m],
                                  "ci_low": lo, "ci_high": hi, "p_bootstrap": p_boot, "p_mcnemar": mc})
    comp = pd.DataFrame(comps)
    comp.round(4).to_csv(out / "comparisons.csv", index=False, encoding="utf-8-sig")

    pr = pd.DataFrame({"text_no": np.arange(1, n + 1), "text": ev[text_col].astype(str),
                       f"{main_short}_3": ev["_ref3"], f"{main_short}_9": ev["_ref9"]})
    if "item" in ev.columns:
        pr.insert(0, "item", ev["item"])
    if w is not None:
        pr["weight"] = w
    for a in arms:
        pr[f"{a}_label"] = [r["label"] for r in preds[a]]
        pr[f"{a}_hate_type"] = [r.get("hate_type") for r in preds[a]]
        pr[f"{a}_reason"] = [r.get("reason", "") for r in preds[a]]
        pr[f"{a}_3"] = p3[a]
        pr[f"{a}_9"] = p9[a]
    pr.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")

    def fmt(v, d=3):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    log(f"\n=== Results against the {main_ref} ===")
    log("  (macro averages over the reference classes; 95% bootstrap intervals in brackets; 'hateful' is the share "
        "of texts labeled hateful)")
    for lev, _ in levels:
        for wname, _ in weightings:
            g = met[(met.reference == main_ref) & (met.level == lev) & (met.weighting == wname)]
            if g.empty:
                continue
            log(f"\n  {lev} class, {wname}")
            log(f"  {'arm':13s} {'macro F1':>8s} {'[95% CI]':>15s} {'kappa':>6s} {'[95% CI]':>15s} {'agree':>6s} "
                f"{'hateful':>7s} {'unspec':>6s}")
            for _, r in g.iterrows():
                extra = f"  sd over sets: F1 {r['sd_over_sets_macro_F1']:.3f}, kappa {r['sd_over_sets_kappa']:.3f}" \
                    if r.arm == "fixed (mean)" else ""
                log(f"  {r.arm:13s} {r.macro_F1:8.3f} [{fmt(r.macro_F1_lo)}, {fmt(r.macro_F1_hi)}] {r.kappa:6.3f} "
                    f"[{fmt(r.kappa_lo)}, {fmt(r.kappa_hi)}] {r.accuracy:6.3f} {r.pred_hateful_share:7.3f} "
                    f"{int(r.unspecified):6d}{extra}")
            ref_row = "(released)"
            log(f"  {ref_row:13s} {'':8s} {'':15s} {'':6s} {'':15s} {'':6s} {g.ref_hateful_share.iloc[0]:7.3f}")

    if not comp.empty:
        log(f"\n=== Paired differences, raw, both arms scored against the {main_ref} ===")
        log("  (difference [95% bootstrap interval], bootstrap p; McNemar p for agreement; reweighted rows in comparisons.csv)")
        for (a, b, lev), g in comp[comp.weighting == "raw"].groupby(["a", "b", "level"], sort=False):
            f, k, ag = (g[g.metric == m].iloc[0] for m in ("macro_F1", "kappa", "accuracy"))
            log(f"  {a} minus {b}, {lev} class: macro F1 {f['diff']:+.3f} [{f.ci_low:+.3f}, {f.ci_high:+.3f}] "
                f"p {pfmt(f.p_bootstrap)}; kappa {k['diff']:+.3f} [{k.ci_low:+.3f}, {k.ci_high:+.3f}] "
                f"p {pfmt(k.p_bootstrap)}; agreement {ag['diff']:+.3f}, McNemar p {pfmt(ag.p_mcnemar)}")

    s2rows, s2boot = [], {}
    if has_types:
        for ref_name, (y3, y9) in refs.items():
            for a in arms:
                res = stage2_scores(y3, y9, p3[a], p9[a], args.n_boot, args.seed + 11, unnamed=unnamed[a])
                if res is None:
                    continue
                s2rows.append({"arm": a, "reference": ref_name, **res[0]})
                s2rows[-1]["agreement_lo"], s2rows[-1]["agreement_hi"] = ci(res[1]["accuracy"])
                s2rows[-1]["kappa_lo"], s2rows[-1]["kappa_hi"] = ci(res[1]["kappa"])
                s2boot[(ref_name, a)] = res[1]
            sets = [a for a in fixed if (ref_name, a) in s2boot]
            if len(sets) > 1:
                g = [r for r in s2rows if r["reference"] == ref_name and r["arm"] in sets]
                row = {"arm": "fixed (mean)", "reference": ref_name, "ref_hateful": g[0]["ref_hateful"]}
                for m in ("both_hateful", "unnamed_as_other", "agreement", "kappa", "macro_F1"):
                    row[m] = float(np.mean([r[m] for r in g]))
                bs = {m: np.mean([s2boot[(ref_name, a)][m] for a in sets], axis=0) for m in ("accuracy", "kappa")}
                row["agreement_lo"], row["agreement_hi"] = ci(bs["accuracy"])
                row["kappa_lo"], row["kappa_hi"] = ci(bs["kappa"])
                s2rows.append(row)
                s2boot[(ref_name, "fixed (mean)")] = bs
    s2main = sorted([r for r in s2rows if r["reference"] == main_ref], key=lambda r: (arm_rank(r["arm"]), r["arm"]))
    if s2main:
        pd.DataFrame(s2rows).round(4).to_csv(out / "stage2.csv", index=False, encoding="utf-8-sig")
        log(f"\n=== Stage 2, as in the paper: hate type on the texts that the {main_ref} and the arm both call hateful ===")
        log("  (agreement over the seven hate types and Cohen's kappa, raw, 95% bootstrap intervals; 'as other' counts "
            "hateful labels whose type names none of the six defined types, scored as other)")
        log(f"  {'arm':13s} {'both hateful':>13s} {'agree':>6s} {'[95% CI]':>15s} {'kappa':>6s} {'[95% CI]':>15s} "
            f"{'as other':>8s}")

        def num(v):
            return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"

        for r in s2main:
            log(f"  {r['arm']:13s} {num(r['both_hateful']):>6s} of {r['ref_hateful']:<3d} {r['agreement']:6.3f} "
                f"[{fmt(r['agreement_lo'])}, {fmt(r['agreement_hi'])}] {r['kappa']:6.3f} "
                f"[{fmt(r['kappa_lo'])}, {fmt(r['kappa_hi'])}] {num(r['unnamed_as_other']):>8s}")
        for a, b in pairs:
            if not a or not b or (main_ref, a) not in s2boot or (main_ref, b) not in s2boot:
                continue
            parts2 = []
            for m, name in (("accuracy", "agreement"), ("kappa", "kappa")):
                d = s2boot[(main_ref, a)][m] - s2boot[(main_ref, b)][m]
                d = d[~np.isnan(d)]
                ra = next(r for r in s2main if r["arm"] == a)
                rb = next(r for r in s2main if r["arm"] == b)
                key = "agreement" if m == "accuracy" else "kappa"
                lo, hi = ci(d)
                p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
                parts2.append(f"{name} {ra[key] - rb[key]:+.3f} [{lo:+.3f}, {hi:+.3f}] p {pfmt(p_boot)}")
            log(f"  {a} minus {b}: " + "; ".join(parts2))


def verify_repo(log, local_path=None):
    import ast
    import inspect
    try:
        if local_path:
            nb = json.loads(Path(local_path).read_text(encoding="utf-8"))
            where = local_path
        else:
            nb = requests.get(NOTEBOOK_RAW, timeout=60).json()
            where = f"GitHub, commit {REPO_COMMIT[:7]}"
        src = "".join(nb["cells"][0]["source"])
        clean = "\n".join(("# " + ln) if ln.lstrip().startswith(("!", "%")) else ln for ln in src.split("\n"))
        tree = ast.parse(clean)
        ns, funcs = {}, {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PROMPT", "examples"):
                exec(compile(ast.Module([node], []), "<notebook>", "exec"), ns)
            if isinstance(node, ast.FunctionDef) and node.name in ("build_enhanced_prompt", "extract_json_enhanced"):
                funcs[node.name] = ast.get_source_segment(clean, node)
        same = {
            "prompt": ns.get("PROMPT") == PROMPT,
            "corpus": ns.get("examples") == examples,
            "prompt builder": funcs.get("build_enhanced_prompt") == inspect.getsource(build_enhanced_prompt).rstrip("\n"),
            "parser": funcs.get("extract_json_enhanced") == inspect.getsource(extract_json_enhanced).rstrip("\n"),
        }
        log(f"  check against {where}: " + ", ".join(f"{k} {'same' if v else 'DIFFERENT'}" for k, v in same.items()))
    except Exception as e:
        log(f"  repository check failed: {str(e)[:200]}")


if __name__ == "__main__":
    main()
