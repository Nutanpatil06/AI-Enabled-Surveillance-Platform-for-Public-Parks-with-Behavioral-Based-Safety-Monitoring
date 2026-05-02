console.log("script.js loaded");

// ---------- LOGIN ----------
async function login() {
    console.log("Login clicked");

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    const msg = document.getElementById("msg");

    const form = new FormData();
    form.append("email", email);
    form.append("password", password);

    const res = await fetch("/login", {
        method: "POST",
        body: form,
        credentials: "include"
    });

    const data = await res.json();
    console.log(data);

    if (data.message) {
        window.location.href = "/dashboard";
    } else {
        msg.innerText = data.error;
    }
}

// ---------- LOGOUT ----------
async function logout() {
    console.log("Logout clicked");

    await fetch("/logout", {
        method: "GET",
        credentials: "include"
    });

    window.location.href = "/";
}

// ---------- UPLOAD ----------
let videoId = null;

async function uploadVideo() {
    console.log("Upload clicked");

    const fileInput = document.getElementById("videoFile");
    const status = document.getElementById("status");

    if (!fileInput.files.length) {
        status.innerText = "Please select a video";
        return;
    }

    const form = new FormData();
    form.append("video", fileInput.files[0]);

    status.innerText = "Uploading...";

    const res = await fetch("/upload-video", {
        method: "POST",
        body: form,
        credentials: "include"
    });

    const data = await res.json();
    console.log(data);

    videoId = data.video_id;
    status.innerText = "Processing...";
    pollProgress();
}

// ---------- PROGRESS ----------
function pollProgress() {
    const bar = document.getElementById("progressFill");

    const interval = setInterval(async () => {
        const res = await fetch(`/progress/${videoId}`, {
            credentials: "include"
        });
        const data = await res.json();

        bar.style.width = data.progress + "%";
        bar.innerText = data.progress + "%";

        if (data.progress >= 100) {
            clearInterval(interval);
            loadAnalytics();
        }
    }, 1000);
}

// ---------- ANALYTICS ----------
async function loadAnalytics() {
    const res = await fetch(`/analytics/${videoId}`, {
        credentials: "include"
    });
    const data = await res.json();

    document.getElementById("auth").innerText = data.authorized;
    document.getElementById("unauth").innerText = data.unauthorized;

    document.getElementById("downloadLink").href =
        `/report/${videoId}`;

    loadLogs();
}

// ---------- LOGS ----------
async function loadLogs() {
    const res = await fetch(`/admin/flagged/${videoId}`, {
        credentials: "include"
    });
    const data = await res.json();

    const logs = document.getElementById("logs");
    logs.innerHTML = "";

    if (!data.length) {
        logs.innerHTML = "<li>No flagged activities</li>";
        return;
    }

    data.forEach(l => {
        logs.innerHTML +=
            `<li>${l.time} - ${l.activity} (${l.confidence})</li>`;
    });
}
