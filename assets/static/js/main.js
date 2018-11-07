$(function () {
    var answer = $("#answer");
    answer.hide();

    $("#send").click(function () {
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

        return false;
    });

    $('#paragraph, #question').bind('input propertychange', function() {
        answer.text("....").hide();
    });
});
