// Retrieve iframe and set the src attribute
$.getJSON(getWebAppBackendUrl('tensorboard-endpoint'), function({ tb_url }) {
    console.log('Received data from backend', tb_url)
    $("#tensorboard-iframe").attr("src", tb_url)
});