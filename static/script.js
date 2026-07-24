function uploadImage() {
    let fileInput = document.getElementById("imageUpload");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.marked_attendance.length > 0) {
            document.getElementById("result").innerText = "Attendance Marked for: " + data.marked_attendance.join(", ");
        } else {
            document.getElementById("result").innerText = "No known faces detected.";
        }
    })
    .catch(error => console.error("Error:", error));
}
