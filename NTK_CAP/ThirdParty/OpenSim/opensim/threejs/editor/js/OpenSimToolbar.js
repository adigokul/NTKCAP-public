import { UIPanel, UIButton, UICheckbox } from './libs/ui.js';

var OpenSimToolbar = function ( editor ) {

	var signals = editor.signals;

	var container = new UIPanel();
	//container.setPaddingTop('10px');
    // This causes 3 columns of buttons:
	container.dom.style.width = '120px';
	container.setId( 'opensim_toolbar' );

	var buttons = new UIPanel();
	container.add( buttons );

	var camera = editor.camera;
    // +X
	function createOpenSimToolbarButton(bTitle, bImage){
		var viewxIcon = document.createElement('img');
		viewxIcon.src = bImage; //;
		var viewx = new UIButton();
		viewx.dom.title = bTitle;
		viewx.dom.className = 'Button';
		viewx.dom.appendChild(viewxIcon);
		return viewx;
	}
	var viewx = createOpenSimToolbarButton('-X', 'icons/frontView_axes.png');
	viewx.onClick(function () {
	    viewfromPlusX();
	});
	buttons.add(viewx);
	function viewfromPlusX() {
	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    //var helper = new THREE.BoundingBoxHelper(modelObject, 0xff0000);
	    //helper.update();
	    // If you want a visible bounding box
	    //editor.scene.add(helper);
	    // create a sphere at new CameraPos
	    var newpos = new THREE.Vector3().copy(center);
	    var fov = editor.camera.fov * (Math.PI / 180);
	    // Calculate the camera distance
	    var objectSize = Math.max(bbox.max.y - bbox.min.y, bbox.max.z - bbox.min.z) / 2;
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    newpos.x = bbox.max.x + distance;
		editor.updateCamera(newpos, center);
		var targetQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler( 0, Math.PI * 0.5, 0 ) );
		editor.signals.moveCameraTo.dispatch(newpos, center, targetQuat);
		//editor.sceneLight.position.copy(editor.camera.position);
    };
    // +Y
	var viewplusy = createOpenSimToolbarButton('-Y', 'icons/topView_axes.png');
	viewplusy.onClick(function () {
	    viewfromPlusY();
	});
	buttons.add(viewplusy);
	function viewfromPlusY() {
	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    var newpos = new THREE.Vector3().copy(center);
	    var fov = editor.camera.fov * (Math.PI / 180);
	    // Calculate the camera distance
	    var objectSize = Math.max(bbox.max.x - bbox.min.x, bbox.max.z - bbox.min.z) / 2;
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    newpos.y = bbox.max.y + distance;
	    editor.updateCamera(newpos, center);
		var targetQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler( - Math.PI * 0.5, 0, 0 ) );
		editor.signals.moveCameraTo.dispatch(newpos, center, targetQuat);
	};
    // +Z
	var viewplusz = createOpenSimToolbarButton('-Z', 'icons/rightView_axes.png');
	viewplusz.onClick(function() {
	    viewfromPlusZ();
	});
	buttons.add(viewplusz);
	function viewfromPlusZ() {

	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    var newpos = new THREE.Vector3().copy(center);
	    var fov = editor.camera.fov * (Math.PI / 180);
	    // Calculate the camera distance
	    var objectSize = Math.max(bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y)/2;
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    newpos.z = bbox.max.z + distance;
		editor.updateCamera(newpos, center);
		var targetQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler() );
		editor.signals.moveCameraTo.dispatch(newpos, center, targetQuat);
	    
    };
    // -X
	var viewminx = createOpenSimToolbarButton('+X', 'icons/backView_axes.png');
	viewminx.onClick(function () {
		viewfromMinusX();
	});
	buttons.add(viewminx);
	function viewfromMinusX() {
	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    var newpos = new THREE.Vector3().copy(center);
	    var fov = editor.camera.fov * (Math.PI / 180);
	    // Calculate the camera distance
	    var objectSize = Math.max(bbox.max.y - bbox.min.y, bbox.max.z - bbox.min.z) / 2;
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    newpos.x = bbox.min.x - distance;
	    editor.updateCamera(newpos, center);
		var targetQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler( 0, -Math.PI * 0.5, 0 ) );
		editor.signals.moveCameraTo.dispatch(newpos, center, targetQuat);	    
    };
    // -Y
	var viewminy = createOpenSimToolbarButton('+Y', 'icons/bottomView_axes.png');
	viewminy.onClick(function () {
	    viewfromMinusY();
	});
	buttons.add(viewminy);
	function viewfromMinusY() {
	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    var newpos = new THREE.Vector3().copy(center);
	    var fov = editor.camera.fov * (Math.PI / 180);
	    // Calculate the camera distance
	    var objectSize = Math.max(bbox.max.x - bbox.min.x, bbox.max.z - bbox.min.z) / 2;
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    newpos.y = bbox.min.y - distance;
		editor.updateCamera(newpos, center);
		var targetQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler( Math.PI * 0.5, 0, 0 ) );
		editor.signals.moveCameraTo.dispatch(newpos, center, targetQuat);
	    
    };
    // -Z
	var viewminz = createOpenSimToolbarButton('+Z', 'icons/leftView_axes.png');
	viewminz.onClick(function () {
	    viewfromMinusZ();
	});

	buttons.add(viewminz);
	function viewfromMinusZ() {

	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    var newpos = new THREE.Vector3().copy(center);
	    var fov = editor.camera.fov * (Math.PI / 180);
	    // Calculate the camera distance
	    var objectSize = Math.max(bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y)/2;
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    newpos.z = bbox.min.z - distance;
		editor.updateCamera(newpos, center);
		var targetQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler(0, Math.PI, 0) );
		editor.signals.moveCameraTo.dispatch(newpos, center, targetQuat);
	    
	};

	var view_zoomout = createOpenSimToolbarButton('Zoom-out', 'icons/zoom-out.png');
	view_zoomout.onClick(function () {
	    editor.viewZoom(-100);
	});
	buttons.add(view_zoomout);
	
	var view_zoomin = createOpenSimToolbarButton('Zoom-in', 'icons/zoom-in.png');
	view_zoomin.onClick(function () {
	    editor.viewZoom(+100);
	});
	buttons.add(view_zoomin);
	var view_refit = createOpenSimToolbarButton('Refit', 'icons/refit.png');
	view_refit.onClick(function () {
	    //var modelObject = editor.scene.getObjectByName('OpenSimModel');
	    var bbox = computeModelBbox();
	    var center = new THREE.Vector3();
	    bbox.getCenter(center);
	    //var newpos = new THREE.Vector3().copy(camera.location);
	    var fov = camera.fov * (Math.PI / 180);
	    var objectSize = Math.max(bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y, bbox.max.z - bbox.min.z);
	    var distance = Math.abs(objectSize / Math.sin(fov / 2));
	    //var helper = new THREE.BoundingBoxHelper(modelObject, 0xff0000);
	    //helper.update();
	    //editor.scene.add(helper);
	    // Zoom out only if model is outside view, also zoom in if too small
	    var curDistance = center.distanceTo(camera.position);
	    editor.viewZoom(curDistance - distance);
	});
	buttons.add(view_refit);

	var snapshot = createOpenSimToolbarButton('Snapshot', 'icons/camera.png');
	snapshot.onClick(function () {
            saveAsImage();
	});
	buttons.add(snapshot);

	function computeModelBbox() {
	    var modelObject = editor.getModel();
	    return (new THREE.Box3().setFromObject(modelObject));
	};
	function saveAsImage() {
		editor.signals.hiresRender.dispatch();
	};
	var recordButton = createOpenSimToolbarButton('Record Start/Stop', 'icons/video.png');
	var recordStatus = false;
	recordButton.onClick(function () {
		recordStatus = !recordStatus;
		if (recordStatus) // Dark red if recording, should be done in CSS
			recordButton.dom.style.backgroundColor = "#FF0808";
		else // default
			recordButton.dom.style.backgroundColor = "";
		toggleRecord();

	});
	buttons.add(recordButton);
	function toggleRecord() {
            editor.toggleRecord();
    }
	var globalFrameButton = createOpenSimToolbarButton('Global Frame X(Red), Y(Green) Z(Blue)', 'icons/axes.png');
	globalFrameButton.onClick(function () {
		editor.toggleGlobalFrame();
	});
	buttons.add(globalFrameButton);
	return container;
}

export { OpenSimToolbar };
