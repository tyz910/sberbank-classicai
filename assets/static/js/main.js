$(function () {
    var answer = $("#answer");
    answer.hide();

    function updateAnswer () {
        answer.text("....").show();

        $.ajax({
            method: "POST",
            url: "/generate/" + $("#poet").val(),
            data: JSON.stringify({
                seed: $("#question").val()
            }),
            contentType : "application/json",
        }).done(function (data) {
            answer.html(data.poem.replace(/\n/g,"<br>"));
        });
    }

    $("#send").click(function () {
        updateAnswer();
        return false;
    });

    $('#question').bind('input propertychange', function() {
        answer.text("....").hide();
    });

    $('#poet').bind('input propertychange', function() {
        updateAnswer();
    });
});
