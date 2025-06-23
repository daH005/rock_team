const mainEl = document.getElementById("js-load-box");
const videoEl = document.getElementById("js-load-box-video");
const inputEl = document.getElementById("js-load-box-input");
const popupEl = document.getElementById("js-popup");

const resultEl = document.getElementById("js-result");
const frameTempEl = document.getElementById("js-frame-temp");
const faceTempEl = document.getElementById("js-face-temp");

mainEl.onclick = () => {
    inputEl.click();
}

inputEl.oninput = async (e) => {
    videoEl.src = URL.createObjectURL(e.target.files[0]);
    resultEl.innerHTML = "";
    popupEl.classList.add("popup--active")

    let r = await fetch("/recognize/onVideo", {
        method: "POST",
        body: e.target.files[0],
    });
    let frames = await r.json();
    console.log(frames);

    let frameEl;
    let facesEl;
    let faceEl;
    let formattedTime;
    let prevFormattedTime = "";
    for (let frame of frames) {

        formattedTime = formatTime(frame[0]);
        if (formattedTime != prevFormattedTime) {
            prevFormattedTime = formattedTime;
            frameEl = frameTempEl.content.cloneNode(true);
            frameEl.querySelector(".frame__time").textContent = formattedTime;
            facesEl = frameEl.querySelector(".faces");
        }

        for (let face of frame[1]) {
            faceEl = faceTempEl.content.cloneNode(true);
            faceEl.querySelector("img").src = "/takeResultImage/" + face[1];
            faceEl.querySelector(".face__name").textContent = face[0];
            facesEl.append(faceEl);
        }
        resultEl.append(frameEl);
    }
    popupEl.classList.remove("popup--active")
}

function formatTime(seconds) {
    const totalSeconds = Math.floor(seconds);

    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;

    const pad = (num) => String(num).padStart(2, '0');

    if (hrs > 0) {
        return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
    } else {
        return `${pad(mins)}:${pad(secs)}`;
    }
}
