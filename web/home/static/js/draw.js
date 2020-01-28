'use strict';

function readTextFile(file, callback) {
  var rawFile = new XMLHttpRequest();
  rawFile.overrideMimeType("application/json");
  rawFile.open("GET", file, true);
  rawFile.onreadystatechange = function() {
      if (rawFile.readyState === 4 && rawFile.status == "200") {
          callback(rawFile.responseText);
      }
  }
  rawFile.send(null);
}

var graph = null, // canvas manager
    labelgraph = null; // label canvas manager
var currentModel = 0; // current model name
var loading = false; // the image is loading (waiting response)
var image = null, // image data
    label = null, // label image data
    latent = null, // latent data
    noise = null; // noise data
var config = null; // config file
var imwidth = 256, // image size
    imheight = 256; // image size
var use_args = false; // [deprecated]
var spinner = new Spinner({ color: '#999' });

var CATEGORY = ['background', 'skin', 'nose', 'eye glasses', 'eye', 'brow', 'ear', 'mouth', 'upper lip', 'lower lip', 'hair', 'hat', 'ear rings', 'necklace', 'neck', 'cloth'];
var CATEGORY_COLORS = ['rgb(0, 0, 0)', 'rgb(128, 0, 0)', 'rgb(0, 128, 0)', 'rgb(128, 128, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(128, 128, 128)', 'rgb(64, 0, 0)', 'rgb(192, 0, 0)', 'rgb(64, 128, 0)', 'rgb(192, 128, 0)', 'rgb(64, 0, 128)', 'rgb(192, 0, 128)', 'rgb(64, 128, 128)', 'rgb(192, 128, 128)'];

var COLORS = [ // drawing colors
  'black',
  'rgb(208, 2, 27)',
  'rgb(245, 166, 35)',
  'rgb(248, 231, 28)',
  'rgb(139, 87, 42)',
  'rgb(126, 211, 33)',
  'white',
  'rgb(226, 238, 244)',
  'rgb(226, 178, 213)',
  'rgb(189, 16, 224)',
  'rgb(74, 144, 226)',
  'rgb(80, 227, 194)',
];

var MODEL_NAMES = [
  'SegGAN (1024px)',
]

Date.prototype.format = function (format) {
  var o = {
    'M+': this.getMonth() + 1, //month
    'd+': this.getDate(), //day
    'h+': this.getHours(), //hour
    'm+': this.getMinutes(), //minute
    's+': this.getSeconds(), //second
    'q+': Math.floor((this.getMonth() + 3) / 3), //quarter
    S: this.getMilliseconds() //millisecond
  };
  if (/(y+)/.test(format)) format = format.replace(RegExp.$1, (this.getFullYear() + '').substr(4 - RegExp.$1.length));
  for (var k in o) {
    if (new RegExp('(' + k + ')').test(format)) format = format.replace(RegExp.$1, RegExp.$1.length == 1 ? o[k] : ('00' + o[k]).substr(('' + o[k]).length));
  }return format;
};

var MAX_LINE_WIDTH = 10;

function setColor(color) {
  graph.setCurrentColor(color);
  $('#color-drop-menu .color-block').css('background-color', color);
  $('#color-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setCategory(color) {
  labelgraph.setCurrentColor(color);
  $('#category-drop-menu .color-block').css('background-color', color);
  $('#category-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setLineWidth(width) {
  graph.setLineWidth(width * 2);
  labelgraph.setLineWidth(width * 2);
  $('#width-label').text(width);
}

function setModel(model) {
  if (loading) return;
  currentModel = model;
  $('#model-label').text(MODEL_NAMES[model]);
  onStart();
}

function setImage(data) {
  setLoading(false);
  if (!data || !data.ok) return;

  if (!image) {
    $('#stroke').removeClass('disabled');
    $('#option-buttons').prop('hidden', false);
    $('#extra-args').prop('hidden', false);
  }

  image = data.img;
  latent = data.latent;
  noise = data.noise;
  label = data.label;
  $('#image').attr('src', image);
  $('#canvas').css('background-image', 'url(' + image + ')');
  $('#label').attr('src', label);
  $('#label-canvas').css('background-image', 'url(' + label + ')');
  graph.setHasImage(true);
  labelgraph.setHasImage(true);
  spinner.spin();
}

function setLoading(isLoading) {
  loading = isLoading;
  graph.setHasImage(false);
  labelgraph.setHasImage(false);
  $('#start-new').prop('disabled', loading);
  $('#submit').prop('disabled', loading);
  $('#spin').prop('hidden', !loading);
  if (loading) {
    $('.image').css('opacity', 0.7);
    spinner.spin(document.getElementById('spin'));
  } else {
    $('.image').css('opacity', 1);
    spinner.stop();
  }
}

function onSubmit() {
  console.log(loading);
  if (graph && !loading) {
    setLoading(true);
    var formData = {
      model: MODEL_NAMES[currentModel],
      image_stroke: graph.getImageData(),
      label_stroke: labelgraph.getImageData(),
      latent: latent,
      noise: noise
    };
    console.log(formData);
    $.post('stroke', formData, setImage, 'json');
  }
}

function onStartNew() {
  if (loading) return;
  setLoading(true);
  graph.clear();
  labelgraph.clear();
  $.post('new', { model: MODEL_NAMES[currentModel] }, setImage, 'json');
}

function onStart() {
  onStartNew();
  $('#start').prop('hidden', true);
}

function onUpload() {
  // filename = $('#choose').val();
  // if (!filename) {
  //   alert('未选择文件');
  //   return false;
  // }
  // $.get(filename, function(data) {
  //   console.log(data);
  // });
  // return false;
}

function onChooseFile(e) {
  // filename = $('#choose').val();
  // console.log(filename);
  // $('.custom-file-control').content(filename);
  // console.log($('.custom-file-control').after());
}

function init() {
  COLORS.forEach(function (color) {
    $('#color-menu').append(
      '\n<li role="presentation">\n  <div onclick="setColor(\'' +
      color +
      '\')"\n  >\n    <div class="color-block" style="background-color:' +
      color + ';border:' +
      (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') +
      '"/>\n  </div>\n</li>');
  });

  CATEGORY_COLORS.forEach(function (color, idx) {
    $('#category-menu').append(
      '\n<li role="presentation">\n' +
      ' <div style="float:left;width:100%" onclick="setCategory(\'' +color + '\')">\n' + 
      '   <div class="color-block" style="float:left;background-color:' + color + 
      ';border:' + (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') + '"/>\n' +
      '   <div class="semantic-block" >' + 
      CATEGORY[idx] + '</div>\n</div>\n</li>');
  });


  MODEL_NAMES.forEach(function (model, idx) {
    $('#model-menu').append(
      '<li role="presentation">\n' +
      '  <div class="dropdown-item model-item" onclick="setModel(' + idx + ')">' +
      model +
      '  </div>\n' +
      '</li>\n'
    );
  });

  var slider = document.getElementById('slider');
  noUiSlider.create(slider, {
    start: 5,
    step: 1,
    range: {
      'min': 1,
      'max': 10
    },
    behaviour: 'drag-tap',
    connect: [true, false],
    orientation: 'vertical',
    direction: 'rtl',
  });
  slider.noUiSlider.on('update', function () {
    setLineWidth(parseInt(slider.noUiSlider.get()));
  });

  setColor('black');
  setLineWidth(5);

  $('#download-sketch').click(function () {
    download(
      canvas.toDataURL('image/png'),
      'sketch_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-image').click(function () {
    download(
      image,
      'image_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#clear-image').click(graph.clear);
  $('#clear-label').click(labelgraph.clear);
  $('#submit').click(onSubmit);
  $('#stroke').click(function() {
    var stroke = $('#stroke').hasClass('active');
    if (stroke) {
      $('#image').prop('hidden', false);
      $('#label').prop('hidden', false);
      $('#canvas').prop('hidden', true);
      $('#label-canvas').prop('hidden', true);
      $('#stroke').removeClass('active');
      $('#stroke .btn-text').text('Show stroke');
    } else {
      $('#image').prop('hidden', true);
      $('#label').prop('hidden', true);
      $('#canvas').prop('hidden', false);
      $('#label-canvas').prop('hidden', false);
      $('#stroke').addClass('active');
      $('#stroke .btn-text').text('Hide stroke');
    }
  });
  $('#start-new').click(onStartNew);
  $('#start').click(onStart);
  $('#add-args').click(onAddArgs);
  // $('#choose').change(onChooseFile);
  // $('#upload').click(onUpload);
}

function download(data, filename) {
  var link = document.createElement('a');
  link.href = data;
  link.download = filename;
  link.click();
}

function onAddArgs() {
  $('#extra-args').prop('hidden', false);
  if (use_args == false) {
    use_args = true;
    $('#args-text').prop('hidden', false);
  } else if (use_args == true) {
    use_args = false;
    $('#args-text').prop('hidden', true);
  }
}

$(document).ready(function () {
  // get config
  readTextFile("static/config.json",
    function(text){
      config = JSON.parse(text);
      MODEL_NAMES = Object.keys(config.models);
      var key = MODEL_NAMES[0];
      imwidth = config.models[key].output_size;
      imheight = config.models[key].output_size;
      document.getElementById('model-label').textContent = key;
      graph = new Graph(document, 'canvas');
      labelgraph = new Graph(document, 'label-canvas');
      var x = document.getElementById('image-container');
      var h = x.clientHeight;
      var w = x.clientWidth;
      var size = h < w ? h : w;
      graph.setSize(size, size);
      labelgraph.setSize(size, size);
      x = document.getElementById('image');
      x.width = size;
      x.height = size;
      x = document.getElementById('label');
      x.width = size;
      x.height = size;
      init();
    });
});
