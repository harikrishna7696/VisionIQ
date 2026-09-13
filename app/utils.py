import yaml
import redis
import io
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

class Config:
    config = yaml.load(open("config/config.yaml"), Loader=yaml.FullLoader)
    YOLO_MODEL = config["YOLO"]["detection_model"]
    CONFIDENCE = config["YOLO"]["confidence"]
    USE_TENSORRT = config["YOLO"]["use_tensorrt"]
    MAX_NEW_TOKENS = config["VLM"]["max_new_tokens"]
    QWEN_MODEL = config["VLM"]["model"]
    CLIP_MODELS = config["EMBEDDING"]["model"]
    GROUNDING_DINO = config["GROUNDING_DINO"]["model"]


class RedisManager:
    def __init__(self, host="localhost", port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=False)

    def set_value(self, key, value,expire_time=None):
        self.redis.set(key, value, ex=expire_time)

    def get_value(self, key):
        return self.redis.get(key)


def image_encoding_using_pillow(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG",quality=100)
    return buffer.getvalue()

def image_decoding_using_pillow(encoded_image):
    image = Image.open(io.BytesIO(encoded_image))
    # Force Pillow to load image data now
    image.load()
    return image

def ai_loader(message="Processing..."):
    placeholder = st.empty()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
            }}

            .container {{
                height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                font-family: Arial, sans-serif;
            }}

            .orb {{
                width: 55px;
                height: 55px;
                border-radius: 50%;

                background:
                    radial-gradient(
                        circle at 35% 35%,
                        white 0%,
                        #00ffff 15%,
                        #0088ff 45%,
                        #001a33 75%
                    );

                box-shadow:
                    0 0 10px #00ffff,
                    0 0 25px #00ffff,
                    0 0 45px #0088ff;

                animation: pulse 1.5s ease-in-out infinite;
                position: relative;
            }}

            .orb::before {{
                content: "";
                position: absolute;

                width: 80px;
                height: 80px;

                left: -14px;
                top: -14px;

                border-radius: 50%;

                border: 2px solid #00ffff;
                border-top-color: transparent;

                animation: rotate 1.2s linear infinite;
            }}

            .orb::after {{
                content: "";
                position: absolute;

                width: 100px;
                height: 100px;

                left: -24px;
                top: -24px;

                border-radius: 50%;

                border: 1px solid #0088ff;
                border-bottom-color: transparent;

                animation: reverseRotate 2s linear infinite;
            }}

            .scan {{
                position: absolute;

                width: 75px;
                height: 2px;

                background: #00ffff;

                box-shadow:
                    0 0 8px #00ffff,
                    0 0 15px #00ffff;

                animation: scan 1.5s ease-in-out infinite;
            }}

            .text {{
                margin-top: 38px;

                color: #00ffff;

                font-size: 14px;

                letter-spacing: 1px;

                text-shadow:
                    0 0 5px #00ffff;
            }}

            .dots::after {{
                content: "";
                animation: dots 1.5s infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{
                    transform: scale(0.85);
                }}

                50% {{
                    transform: scale(1.1);
                }}
            }}

            @keyframes rotate {{
                from {{
                    transform: rotate(0deg);
                }}

                to {{
                    transform: rotate(360deg);
                }}
            }}

            @keyframes reverseRotate {{
                from {{
                    transform: rotate(360deg);
                }}

                to {{
                    transform: rotate(0deg);
                }}
            }}

            @keyframes scan {{
                0% {{
                    transform: translateY(-25px);
                    opacity: 0;
                }}

                30% {{
                    opacity: 1;
                }}

                70% {{
                    opacity: 1;
                }}

                100% {{
                    transform: translateY(25px);
                    opacity: 0;
                }}
            }}

            @keyframes dots {{
                0% {{
                    content: "";
                }}

                25% {{
                    content: ".";
                }}

                50% {{
                    content: "..";
                }}

                75% {{
                    content: "...";
                }}
            }}
        </style>
    </head>

    <body>

        <div class="container">

            <div style="position: relative;">

                <div class="orb"></div>

                <div class="scan"></div>

            </div>

            <div class="text">
                {message}<span class="dots"></span>
            </div>

        </div>

    </body>
    </html>
    """

    with placeholder:
        components.html(
            html,
            height=190,
            scrolling=False
        )

    return placeholder

def object_detection_loader(
    message="Detecting objects...",
    height=320
):
    placeholder = st.empty()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>

    <style>

        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
        }}

        .wrapper {{
            height: {height}px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
        }}

        /* CAMERA FRAME */

        .camera {{
            position: relative;

            width: 300px;
            height: 190px;

            background:
                linear-gradient(
                    rgba(0,255,255,0.025) 1px,
                    transparent 1px
                ),
                linear-gradient(
                    90deg,
                    rgba(0,255,255,0.025) 1px,
                    transparent 1px
                );

            background-size: 20px 20px;

            border: 1px solid rgba(0,255,255,0.25);

            box-shadow:
                0 0 25px rgba(0,255,255,0.08);

            overflow: hidden;
        }}


        /* CAMERA CORNERS */

        .corner {{
            position: absolute;
            width: 18px;
            height: 18px;

            border-color: #00ffff;
            border-style: solid;

            opacity: 0.9;
        }}

        .tl {{
            top: 0;
            left: 0;
            border-width: 2px 0 0 2px;
        }}

        .tr {{
            top: 0;
            right: 0;
            border-width: 2px 2px 0 0;
        }}

        .bl {{
            bottom: 0;
            left: 0;
            border-width: 0 0 2px 2px;
        }}

        .br {{
            bottom: 0;
            right: 0;
            border-width: 0 2px 2px 0;
        }}


        /* SCANNING LINE */

        .scanner {{
            position: absolute;

            left: 0;
            width: 100%;
            height: 2px;

            background: #00ffff;

            box-shadow:
                0 0 8px #00ffff,
                0 0 18px #00ffff;

            animation: scanning 2.5s linear infinite;
        }}

        @keyframes scanning {{

            0% {{
                top: 0;
                opacity: 0;
            }}

            10% {{
                opacity: 1;
            }}

            90% {{
                opacity: 1;
            }}

            100% {{
                top: 100%;
                opacity: 0;
            }}
        }}


        /* DETECTION BOX */

        .box {{
            position: absolute;

            border: 1.5px solid #00ffff;

            box-shadow:
                0 0 8px rgba(0,255,255,0.4);

            animation: detect 2.5s ease-in-out infinite;
        }}

        .box span {{
            position: absolute;

            top: -20px;
            left: -1px;

            background: #00ffff;

            color: #001015;

            font-size: 10px;

            font-weight: bold;

            padding: 3px 5px;

            white-space: nowrap;
        }}


        /* BOX 1 */

        .box1 {{
            width: 65px;
            height: 90px;

            left: 45px;
            top: 55px;

            animation-delay: 0s;
        }}


        /* BOX 2 */

        .box2 {{
            width: 95px;
            height: 55px;

            right: 40px;
            top: 80px;

            animation-delay: 0.8s;
        }}


        /* BOX 3 */

        .box3 {{
            width: 45px;
            height: 45px;

            left: 145px;
            top: 30px;

            animation-delay: 1.5s;
        }}


        @keyframes detect {{

            0%, 100% {{
                opacity: 0.25;
                transform: scale(0.96);
            }}

            20%, 70% {{
                opacity: 1;
                transform: scale(1);
            }}
        }}


        /* DETECTION DOT */

        .point {{
            position: absolute;

            width: 5px;
            height: 5px;

            border-radius: 50%;

            background: #00ffff;

            box-shadow:
                0 0 8px #00ffff;

            animation: pointPulse 1s infinite;
        }}

        .point1 {{
            left: 45px;
            top: 55px;
        }}

        .point2 {{
            right: 40px;
            top: 80px;
            animation-delay: .5s;
        }}

        @keyframes pointPulse {{

            0%, 100% {{
                transform: scale(1);
                opacity: .5;
            }}

            50% {{
                transform: scale(2);
                opacity: 1;
            }}
        }}


        /* STATUS */

        .status {{
            margin-top: 22px;

            color: #00ffff;

            font-size: 14px;

            letter-spacing: 1px;

            text-shadow:
                0 0 8px rgba(0,255,255,.7);
        }}


        .status::after {{
            content: "";

            animation: dots 1.5s infinite;
        }}

        @keyframes dots {{

            0% {{
                content: "";
            }}

            33% {{
                content: ".";
            }}

            66% {{
                content: "..";
            }}

            100% {{
                content: "...";
            }}
        }}


        /* LIVE INDICATOR */

        .live {{
            position: absolute;

            top: 8px;
            right: 10px;

            font-size: 9px;

            color: #00ffff;

            letter-spacing: 1px;
        }}

        .live-dot {{
            display: inline-block;

            width: 6px;
            height: 6px;

            margin-right: 4px;

            border-radius: 50%;

            background: #00ffff;

            box-shadow: 0 0 8px #00ffff;

            animation: livePulse 1s infinite;
        }}

        @keyframes livePulse {{

            0%, 100% {{
                opacity: .4;
            }}

            50% {{
                opacity: 1;
            }}
        }}

    </style>

    </head>

    <body>

        <div class="wrapper">

            <div class="camera">

                <!-- corners -->

                <div class="corner tl"></div>
                <div class="corner tr"></div>
                <div class="corner bl"></div>
                <div class="corner br"></div>

                <!-- live -->

                <div class="live">
                    <span class="live-dot"></span>
                    AI VISION
                </div>

                <!-- scanning -->

                <div class="scanner"></div>

                <!-- detection points -->

                <div class="point point1"></div>
                <div class="point point2"></div>

                <!-- detection boxes -->

                <div class="box box1">
                    <span>PERSON 98%</span>
                </div>

                <div class="box box2">
                    <span>CAR 94%</span>
                </div>

                <div class="box box3">
                    <span>OBJECT 87%</span>
                </div>

            </div>

            <div class="status">
                {message}<span></span>
            </div>

        </div>

    </body>

    </html>
    """

    with placeholder:

        components.html(
            html,
            height=height,
            scrolling=False
        )

    return placeholder

def futuristic_ai_loader(message="Analyzing video...", height=360):
    placeholder = st.empty()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">

    <style>

    * {{
        box-sizing: border-box;
    }}

    html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        overflow: hidden;
    }}

    .scene {{
        width: 100%;
        height: {height}px;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        font-family: Arial, sans-serif;
    }}

    /* =====================================================
       AI CORE
    ===================================================== */

    .core-container {{
        position: relative;
        width: 220px;
        height: 220px;
    }}

    .core {{
        position: absolute;

        left: 50%;
        top: 50%;

        width: 38px;
        height: 38px;

        transform: translate(-50%, -50%);

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                white 0%,
                #00ffff 15%,
                #008cff 45%,
                rgba(0,100,255,.2) 70%,
                transparent 75%
            );

        box-shadow:
            0 0 10px #00ffff,
            0 0 25px #00ffff,
            0 0 50px #008cff,
            0 0 90px rgba(0,150,255,.8);

        animation: corePulse 1.4s ease-in-out infinite;
    }}


    /* =====================================================
       ROTATING HUD RINGS
    ===================================================== */

    .ring {{
        position: absolute;

        left: 50%;
        top: 50%;

        border-radius: 50%;

        transform: translate(-50%, -50%);

        border: 1px solid rgba(0,255,255,.25);
    }}

    .ring1 {{
        width: 75px;
        height: 75px;

        border-top: 2px solid #00ffff;
        border-right-color: transparent;

        animation: rotate 2s linear infinite;
    }}

    .ring2 {{
        width: 110px;
        height: 110px;

        border-bottom: 2px solid #008cff;
        border-left-color: transparent;

        animation: rotateReverse 3s linear infinite;
    }}

    .ring3 {{
        width: 155px;
        height: 155px;

        border-top: 1px solid #00ffff;
        border-bottom: 1px solid rgba(0,255,255,.1);

        animation: rotate 6s linear infinite;
    }}

    .ring4 {{
        width: 200px;
        height: 200px;

        border-left: 1px solid rgba(0,150,255,.4);
        border-right: 1px solid rgba(0,150,255,.1);

        animation: rotateReverse 10s linear infinite;
    }}


    /* =====================================================
       RADAR SWEEP
    ===================================================== */

    .radar {{
        position: absolute;

        left: 50%;
        top: 50%;

        width: 190px;
        height: 190px;

        transform: translate(-50%, -50%);

        border-radius: 50%;

        overflow: hidden;

        opacity: .7;
    }}

    .radar::after {{
        content: "";

        position: absolute;

        left: 50%;
        top: 50%;

        width: 50%;
        height: 2px;

        transform-origin: left center;

        background:
            linear-gradient(
                90deg,
                rgba(0,255,255,1),
                transparent
            );

        box-shadow: 0 0 10px #00ffff;

        animation: radarSweep 2.2s linear infinite;
    }}


    /* =====================================================
       ENERGY WAVE
    ===================================================== */

    .wave {{
        position: absolute;

        left: 50%;
        top: 50%;

        width: 45px;
        height: 45px;

        transform: translate(-50%, -50%);

        border: 1px solid #00ffff;

        border-radius: 50%;

        opacity: 0;

        animation: wave 2s ease-out infinite;
    }}

    .wave2 {{
        animation-delay: .65s;
    }}

    .wave3 {{
        animation-delay: 1.3s;
    }}


    /* =====================================================
       FLOATING AI PARTICLES
    ===================================================== */

    .particle {{
        position: absolute;

        width: 4px;
        height: 4px;

        border-radius: 50%;

        background: #00ffff;

        box-shadow:
            0 0 8px #00ffff,
            0 0 15px #008cff;
    }}

    .p1 {{
        left: 15px;
        top: 50px;
        animation: particle1 3s infinite;
    }}

    .p2 {{
        left: 190px;
        top: 70px;
        animation: particle2 3.5s infinite;
    }}

    .p3 {{
        left: 35px;
        top: 170px;
        animation: particle3 4s infinite;
    }}

    .p4 {{
        left: 175px;
        top: 160px;
        animation: particle4 2.8s infinite;
    }}

    .p5 {{
        left: 100px;
        top: 5px;
        animation: particle5 3.2s infinite;
    }}

    .p6 {{
        left: 105px;
        top: 205px;
        animation: particle6 3.7s infinite;
    }}


    /* =====================================================
       DATA STREAMS
    ===================================================== */

    .stream {{
        position: absolute;

        width: 35px;
        height: 1px;

        background: linear-gradient(
            90deg,
            transparent,
            #00ffff,
            transparent
        );

        box-shadow: 0 0 8px #00ffff;

        opacity: 0;

        animation: stream 2s linear infinite;
    }}

    .s1 {{
        left: 0;
        top: 100px;
        animation-delay: .2s;
    }}

    .s2 {{
        right: 0;
        top: 120px;
        animation-delay: .8s;
    }}

    .s3 {{
        left: 70px;
        top: 0;
        transform: rotate(90deg);
        animation-delay: 1.2s;
    }}

    .s4 {{
        right: 70px;
        bottom: 0;
        transform: rotate(90deg);
        animation-delay: 1.6s;
    }}


    /* =====================================================
       STATUS
    ===================================================== */

    .status {{
        margin-top: -5px;

        color: #00ffff;

        font-family: monospace;

        font-size: 14px;

        letter-spacing: 2px;

        text-shadow:
            0 0 5px #00ffff,
            0 0 15px rgba(0,255,255,.5);
    }}

    .status::after {{
        content: "";

        animation: dots 1.4s steps(4) infinite;
    }}


    /* =====================================================
       ANIMATIONS
    ===================================================== */

    @keyframes corePulse {{

        0%,100% {{
            transform: translate(-50%,-50%) scale(.8);
            opacity: .75;
        }}

        50% {{
            transform: translate(-50%,-50%) scale(1.15);
            opacity: 1;
        }}
    }}

    @keyframes rotate {{

        from {{
            transform: translate(-50%,-50%) rotate(0deg);
        }}

        to {{
            transform: translate(-50%,-50%) rotate(360deg);
        }}
    }}

    @keyframes rotateReverse {{

        from {{
            transform: translate(-50%,-50%) rotate(360deg);
        }}

        to {{
            transform: translate(-50%,-50%) rotate(0deg);
        }}
    }}

    @keyframes radarSweep {{

        from {{
            transform: rotate(0deg);
        }}

        to {{
            transform: rotate(360deg);
        }}
    }}

    @keyframes wave {{

        0% {{
            width: 40px;
            height: 40px;
            opacity: .8;
        }}

        100% {{
            width: 190px;
            height: 190px;
            opacity: 0;
        }}
    }}

    @keyframes dots {{

        0% {{
            content: "";
        }}

        25% {{
            content: ".";
        }}

        50% {{
            content: "..";
        }}

        75% {{
            content: "...";
        }}
    }}

    @keyframes particle1 {{

        0% {{
            transform: translate(0,0);
            opacity: .2;
        }}

        50% {{
            transform: translate(80px,60px);
            opacity: 1;
        }}

        100% {{
            transform: translate(105px,60px);
            opacity: 0;
        }}
    }}

    @keyframes particle2 {{

        0% {{
            transform: translate(0,0);
            opacity: 0;
        }}

        50% {{
            transform: translate(-75px,45px);
            opacity: 1;
        }}

        100% {{
            transform: translate(-100px,45px);
            opacity: 0;
        }}
    }}

    @keyframes particle3 {{

        0% {{
            transform: translate(0,0);
            opacity: 0;
        }}

        50% {{
            transform: translate(70px,-65px);
            opacity: 1;
        }}

        100% {{
            transform: translate(90px,-75px);
            opacity: 0;
        }}
    }}

    @keyframes particle4 {{

        0% {{
            transform: translate(0,0);
            opacity: 0;
        }}

        50% {{
            transform: translate(-70px,-50px);
            opacity: 1;
        }}

        100% {{
            transform: translate(-90px,-60px);
            opacity: 0;
        }}
    }}

    @keyframes particle5 {{

        0% {{
            transform: translate(0,0);
            opacity: 0;
        }}

        50% {{
            transform: translate(0,80px);
            opacity: 1;
        }}

        100% {{
            transform: translate(0,105px);
            opacity: 0;
        }}
    }}

    @keyframes particle6 {{

        0% {{
            transform: translate(0,0);
            opacity: 0;
        }}

        50% {{
            transform: translate(0,-80px);
            opacity: 1;
        }}

        100% {{
            transform: translate(0,-105px);
            opacity: 0;
        }}
    }}

    @keyframes stream {{

        0% {{
            transform: translateX(-30px);
            opacity: 0;
        }}

        30% {{
            opacity: 1;
        }}

        100% {{
            transform: translateX(40px);
            opacity: 0;
        }}
    }}

    </style>
    </head>

    <body>

    <div class="scene">

        <div class="core-container">

            <!-- rotating HUD -->
            <div class="ring ring4"></div>
            <div class="ring ring3"></div>
            <div class="ring ring2"></div>
            <div class="ring ring1"></div>

            <!-- radar -->
            <div class="radar"></div>

            <!-- energy waves -->
            <div class="wave"></div>
            <div class="wave wave2"></div>
            <div class="wave wave3"></div>

            <!-- AI core -->
            <div class="core"></div>

            <!-- particles -->
            <div class="particle p1"></div>
            <div class="particle p2"></div>
            <div class="particle p3"></div>
            <div class="particle p4"></div>
            <div class="particle p5"></div>
            <div class="particle p6"></div>

            <!-- data streams -->
            <div class="stream s1"></div>
            <div class="stream s2"></div>
            <div class="stream s3"></div>
            <div class="stream s4"></div>

        </div>

        <div class="status">
            {message}
        </div>

    </div>

    </body>
    </html>
    """

    with placeholder:
        components.html(
            html,
            height=height,
            scrolling=False
        )

    return placeholder
