'use strict';

function Graph(document) {
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');

  var width = canvas.width;
  var height = canvas.height;
  var offsetX = canvas.offsetLeft + parseInt(canvas.style.borderBottomWidth);
  var offsetY = canvas.offsetTop + parseInt(canvas.style.borderBottomWidth);

  ctx.fillStyle = '#000';
  ctx.lineCap = 'round';
  ctx.lineWidth = 5;
  
  this.getImageData = function () {
    return canvas.toDataURL();
  };

  this.setCurrentColor = function (colorString) {
    console.log(document.getElementById('canvas').getContext('2d').lineWidth);
    ctx.strokeStyle = colorString;
  };

  this.setLineWidth = function (width) {
    ctx.lineWidth = width;
    console.log(document.getElementById('canvas').getContext('2d').lineWidth);
  };

  this.drawLine = function (x1, y1, x2, y2) {
    console.log(document.getElementById('canvas').getContext('2d').lineWidth);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };

  this.clear = function () {
    console.log(document.getElementById('canvas').getContext('2d').lineWidth);
    ctx.clearRect(0, 0, width, height);
  };

  console.log(document.getElementById('canvas').getContext('2d').lineWidth);
}