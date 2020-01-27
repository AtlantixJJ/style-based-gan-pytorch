'use strict';

function Graph(document, name) {
  var canvas = document.getElementById(name);
  var ctx = canvas.getContext('2d');

  var width = canvas.width;
  var height = canvas.height;
  var offsetX = canvas.offsetLeft + parseInt(canvas.style.borderBottomWidth);
  var offsetY = canvas.offsetTop + parseInt(canvas.style.borderBottomWidth);

  ctx.fillStyle = '#000';
  ctx.lineCap = 'round';
  ctx.lineWidth = 5;

  var mouseDown = false;
  var prev = null;
  var has_image = false;

  canvas.addEventListener('mousedown', this.onMouseDown, false);
  canvas.addEventListener('mouseup', this.onMouseUp, false);
  canvas.addEventListener('mousemove', this.onMouseMove, false);
  canvas.addEventListener('mouseout', this.onMouseUp, false);
  canvas.addEventListener('touchstart', this.onMouseDown, false);
  canvas.addEventListener('touchend', this.onMouseUp, false);
  canvas.addEventListener('touchmove', this.onMouseMove, false);
  canvas.addEventListener('touchcancel', this.onMouseUp, false);

  function getMouse(event) {
    var rect = event.target.getBoundingClientRect();
    var mouse = {
      x: (event.touches && event.touches[0] || event).clientX - rect.left,
      y: (event.touches && event.touches[0] || event).clientY - rect.top
    };
    if (mouse.x > rect.width || mouse.x < 0 || mouse.y > rect.height || mouse.y < 0)
      return null;
    return mouse;
  }

  this.onMouseDown = function (event) {
    if (event.button == 2 || !has_image) {
      mouseDown = false;
      return;
    }
    if (mouseDown) { // double down, basically impossible
      this.onMouseUp(event);
      return;
    }
    prev = getMouse(event);
    if (prev != null) mouseDown = true;
  }
  
  this.onMouseUp = function (event) {
    mouseDown = false;
  }

  this.mouseMove = function(event) {
    event.preventDefault();
    if (mouseDown && !has_image) {
      var mouse = getMouse(event);
      if (mouse == null) {
        mouseDown = false;
        return;
      }
      if (!has_image) return;
      this.drawLine(prev.x, prev.y, mouse.x, mouse.y);
      prev = mouse;
    }
  }

  this.setSize = funtion (height, width) {
    canvas.width = width;
    canvas.height = height;
  }

  this.getImageData = function () {
    return canvas.toDataURL();
  };

  this.setCurrentColor = function (colorString) {
    ctx.strokeStyle = colorString;
  };

  this.setLineWidth = function (width) {
    ctx.lineWidth = width;
  };

  this.drawLine = function (x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };

  this.clear = function () {
    ctx.clearRect(0, 0, width, height);
  };

}