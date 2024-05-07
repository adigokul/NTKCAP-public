/**
 * @author Ayman Habib based on mrdoob / http://mrdoob.com/
 */
 import * as THREE from '../../build/three.module.js';

 import { TransformControls } from '../../examples/jsm/controls/TransformControls.js';
 
 import { UIPanel } from './libs/ui.js';
 
 import { EditorControls } from './EditorControls.js';
 
 import { ViewportCamera } from './Viewport.Camera.js';
 import { ViewportInfo } from './Viewport.Info.js';
 import { ViewHelper } from './Viewport.ViewHelper.js';
 import { VR } from './Viewport.VR.js';
 
 import { SetPositionCommand } from './commands/SetPositionCommand.js';
 import { SetRotationCommand } from './commands/SetRotationCommand.js';
 import { SetScaleCommand } from './commands/SetScaleCommand.js';
 import { sendText } from '../websocket.js';

function OpenSimViewport ( editor ) {

	var signals = editor.signals;

	var container = new UIPanel();
	container.setId( 'viewport' );
	container.setPosition( 'absolute' );

	//container.add( new ViewportInfo( editor ) );

	var camera = editor.camera;
	var scene = editor.scene;
	var sceneHelpers = editor.sceneHelpers;
	var sceneOrtho = editor.sceneOrtho;
	var sceneOrthoCam = editor.sceneOrthoCam;
        
	//var showHelpers = editor.showDebug();
	var dollyCameraEye = editor.cameraEye;
    // Animation related
	var animationCycleTime;
	var animationLookAt = new THREE.Vector3(0, 0, 0);
	var lookAtObject = scene;
	var showCamOnly = false;
    // AnimationRecording
	var capturer = undefined;
	var recording = false;
	var screenCapUpsamplingFactor = 2;
	var objects = [];

	// helpers

	var grid = new THREE.GridHelper( 30, 30, 0x444444, 0x888888 );
	var viewHelper = new ViewHelper( camera, container );
	//OPENSIM sceneHelpers.add( grid );

	//

	// var camera = editor.camera;
	var dollyCamera = editor.dolly_camera;
	//
	var clearColor = editor.config.getKey('settings/backgroundcolor');

	var box = new THREE.Box3();
	var selectionBox = new THREE.BoxHelper();
	selectionBox.material.depthTest = false;
	selectionBox.material.transparent = true;
	selectionBox.visible = false;
	sceneHelpers.add( selectionBox );

	var objectPositionOnDown = null;
	var objectRotationOnDown = null;
	var objectScaleOnDown = null;

	var transformControls = new TransformControls(camera, container.dom);
	editor.control = transformControls;

	transformControls.addEventListener( 'change', function () {

		var object = transformControls.object;

		if ( object !== undefined ) {

			selectionBox.setFromObject( object );

			if ( editor.helpers[ object.id ] !== undefined ) {

				editor.helpers[ object.id ].update();

			}

			signals.refreshSidebarObject3D.dispatch( object );

		}

 		render();

	} );
	transformControls.addEventListener( 'mouseDown', function () {

		var object = transformControls.object;

		objectPositionOnDown = object.position.clone();
		objectRotationOnDown = object.rotation.clone();
		objectScaleOnDown = object.scale.clone();

		controls.enabled = false;

	} );
	transformControls.addEventListener( 'mouseUp', function () {

		var object = transformControls.object;

		if ( object !== undefined ) {

			switch ( transformControls.getMode() ) {

				case 'translate':

					if ( ! objectPositionOnDown.equals( object.position ) ) {

						editor.execute( new SetPositionCommand( editor, object, object.position, objectPositionOnDown ) );
						//
						var json = JSON.stringify({
							"event": "translate",
							"uuid": object.uuid,
							"location": object.position
						});
						sendText(json);
					}

					break;

				case 'rotate':

					if ( ! objectRotationOnDown.equals( object.rotation ) ) {

						editor.execute( new SetRotationCommand( editor, object, object.rotation, objectRotationOnDown ) );

					}

					break;

				case 'scale':

					if ( ! objectScaleOnDown.equals( object.scale ) ) {

						editor.execute( new SetScaleCommand( object, object.scale, objectScaleOnDown ) );

					}

					break;

			}

		}

		controls.enabled = true;

	} );

	sceneHelpers.add( transformControls );

	// fog

	var oldFogType = "None";
	var oldFogColor = 0xaaaaaa;
	var oldFogNear = 1;
	var oldFogFar = 5000;
	var oldFogDensity = 0.00025;

	// object picking

	var raycaster = new THREE.Raycaster();
	var mouse = new THREE.Vector2();

	// events

	function updateAspectRatio() {

		camera.aspect = container.dom.offsetWidth / container.dom.offsetHeight;
		camera.updateProjectionMatrix();

	}

	function getIntersects( point, objects ) {

		mouse.set( ( point.x * 2 ) - 1, - ( point.y * 2 ) + 1 );

		raycaster.setFromCamera( mouse, camera );

		return raycaster.intersectObjects( objects );

	}

	var onDownPosition = new THREE.Vector2();
	var onUpPosition = new THREE.Vector2();
	var onDoubleClickPosition = new THREE.Vector2();

	function getMousePosition( dom, x, y ) {

		var rect = dom.getBoundingClientRect();
		return [ ( x - rect.left ) / rect.width, ( y - rect.top ) / rect.height ];

	}

	function handleClick() {

		if ( onDownPosition.distanceTo( onUpPosition ) === 0 ) {

			var intersects = getIntersects( onUpPosition, objects );

			if ( intersects.length > 0 ) {

				var object = intersects[ 0 ].object;

				if ( object.userData.object !== undefined ) {

					// helper

					editor.select( object.userData.object );

				} else {

					editor.select( object );

				}

			} else {

				editor.select( null );

			}

			render();

		}

	}

	function onMouseDown( event ) {

		event.preventDefault();

		var array = getMousePosition( container.dom, event.clientX, event.clientY );
		onDownPosition.fromArray( array );

		document.addEventListener( 'mouseup', onMouseUp, false );

	}

	function onMouseUp( event ) {

		var array = getMousePosition( container.dom, event.clientX, event.clientY );
		onUpPosition.fromArray( array );

		handleClick();

		document.removeEventListener( 'mouseup', onMouseUp, false );

	}

	function onTouchStart( event ) {

		var touch = event.changedTouches[ 0 ];

		var array = getMousePosition( container.dom, touch.clientX, touch.clientY );
		onDownPosition.fromArray( array );

		document.addEventListener( 'touchend', onTouchEnd, false );

	}

	function onTouchEnd( event ) {

		var touch = event.changedTouches[ 0 ];

		var array = getMousePosition( container.dom, touch.clientX, touch.clientY );
		onUpPosition.fromArray( array );

		handleClick();

		document.removeEventListener( 'touchend', onTouchEnd, false );

	}

	function onDoubleClick( event ) {

		var array = getMousePosition( container.dom, event.clientX, event.clientY );
		onDoubleClickPosition.fromArray( array );

		var intersects = getIntersects( onDoubleClickPosition, objects );

		if ( intersects.length > 0 ) {

			var intersect = intersects[ 0 ];

			signals.objectFocused.dispatch( intersect.object );

		}

	}

	container.dom.addEventListener( 'mousedown', onMouseDown, false );
	container.dom.addEventListener( 'touchstart', onTouchStart, false );
	container.dom.addEventListener( 'dblclick', onDoubleClick, false );

	// controls need to be added *after* main logic,
	// otherwise controls.enabled doesn't work.

	var controls = new EditorControls( camera, container.dom );
	controls.addEventListener( 'change', function () {

		signals.cameraChanged.dispatch( camera );

	} );
	viewHelper.controls = controls;

	// signals

	signals.editorCleared.add( function () {

		controls.center.set( 0, 0, 0 );
		render();

	} );

	signals.renderDebugChanged.add(function (show) {
	    sceneHelpers.visible = show;
	    render();
	});

	signals.themeChanged.add( function ( value ) {

		switch ( value ) {

			case 'css/light.css':
				//OPENSIM sceneHelpers.remove( grid );
				//OPENSIM grid = new THREE.GridHelper( 30, 1, 0x444444, 0x888888 );
				//OPENSIM sceneHelpers.add( grid );
				clearColor = 0xaaaaaa;
				break;
			case 'css/dark.css':
				//OPENSIM sceneHelpers.remove( grid );
				//OPENSIM grid = new THREE.GridHelper( 30, 1, 0xbbbbbb, 0x888888 );
				//OPENSIM sceneHelpers.add( grid );
				clearColor = 0x333333;
				break;
			case 'css/minimal.css':
			    //clearColor = 0xf0f5f5;
				break;

		}

		//renderer.setClearColor( clearColor );

		render();

	} );
	signals.backgroundColorChanged.add(function (newColor) {
		renderer.setClearColor(newColor);
		clearColor = newColor;
		editor.config.setKey('settings/backgroundcolor', newColor);
	    render();
	});
	signals.transformModeChanged.add( function ( mode ) {

		transformControls.setMode( mode );

	} );

	signals.snapChanged.add( function ( dist ) {

		transformControls.setTranslationSnap( dist );

	} );

	signals.spaceChanged.add( function ( space ) {

		transformControls.setSpace( space );

	} );

	signals.rendererChanged.add( function ( newRenderer ) {

		if ( renderer !== null ) {

			container.dom.removeChild( renderer.domElement );

		}

		renderer = newRenderer;

		renderer.autoClear = false;
		renderer.autoUpdateScene = false;
		renderer.setAnimationLoop( animate );
		renderer.setClearColor( clearColor );
		renderer.setPixelRatio( window.devicePixelRatio );
		renderer.setSize( container.dom.offsetWidth, container.dom.offsetHeight );

		container.dom.appendChild( renderer.domElement );

		render();

	} );

	signals.sceneGraphChanged.add( function () {

		render();

	});

	signals.defaultCameraApplied.add(function (newCenter) {
	    controls.center.copy(newCenter);
	});

	var saveTimeout;

	signals.cameraChanged.add( function (camera, offset) {
		editor.sceneLight.position.copy(editor.camera.position);
		if (offset !== undefined)
			controls.center.add (offset);
		render();

	} );

	signals.recordingStarted.add(function () {
	    // add frame to capture
	    if (capturer !== undefined) 
               capturer = undefined;
         capturer = new CCapture({
	            verbose: false,
	            display: false,
	            framerate: 30,
                name: "opensim_video",
                format: 'webm-mediarecorder',
	        });
	        recording = true;
	        capturer.start();
	});

	signals.recordingStopped.add(function () {
	    if (capturer !== undefined) {
	        capturer.stop();
	        capturer.save();
	        capturer = undefined;
            recording = false;
	    }
	    // add frame to gif
	    //gif.render();
	    //gif.finishRendering();
	});

	signals.objectSelected.add(function (object) {

		selectionBox.visible = false;
		transformControls.detach();

		if ( object !== null && object !== scene && object !== camera ) {

			box.setFromObject( object );

			if ( box.isEmpty() === false ) {

				selectionBox.setFromObject( object );
				selectionBox.visible = true;

			}
            if (object.userData !== "NonEditable")
			    transformControls.attach( object );

		}

		render();

	} );

	signals.objectFocused.add( function ( object ) {

		controls.focus( object );

	} );

	signals.geometryChanged.add( function ( object ) {

		if ( object !== undefined ) {

			selectionBox.setFromObject( object );

		}

		render();

	} );

	signals.objectAdded.add( function ( object ) {

		object.traverse( function ( child ) {

			objects.push( child );

		} );

	} );

	signals.objectChanged.add( function ( object ) {

	    if (object === null) return;
	    if (editor.selected === object) {

			selectionBox.setFromObject( object );

		}

		if ( object.isPerspectiveCamera ) {

			object.updateProjectionMatrix();

		}

		if ( editor.helpers[ object.id ] !== undefined ) {

			editor.helpers[ object.id ].update();

		}
		render();

	} );

	signals.objectRemoved.add( function ( object ) {

		object.traverse( function ( child ) {

			objects.splice( objects.indexOf( child ), 1 );

		} );

	} );

	signals.helperAdded.add( function ( object ) {

		objects.push( object.getObjectByName( 'picker' ) );

	} );

	signals.helperRemoved.add( function ( object ) {

		var picker = object.getObjectByName( 'picker' );

		if ( picker !== undefined ) {

			objects.splice( objects.indexOf( picker ), 1 );

		}

	} );

	signals.materialChanged.add( function ( material ) {

		render();

	} );

	signals.fogTypeChanged.add( function ( fogType ) {

		if ( fogType !== oldFogType ) {

			if ( fogType === "None" ) {

				scene.fog = null;

			} else if ( fogType === "Fog" ) {

				scene.fog = new THREE.Fog( oldFogColor, oldFogNear, oldFogFar );

			} else if ( fogType === "FogExp2" ) {

				scene.fog = new THREE.FogExp2( oldFogColor, oldFogDensity );

			}

			oldFogType = fogType;

		}

		render();

	} );

	signals.fogColorChanged.add( function ( fogColor ) {

		oldFogColor = fogColor;

		updateFog( scene );

		render();

	} );

	signals.fogParametersChanged.add( function ( near, far, density ) {

		oldFogNear = near;
		oldFogFar = far;
		oldFogDensity = density;

		updateFog( scene );

		render();

	} );

	signals.windowResize.add( function () {

		// TODO: Move this out?

		editor.camera.aspect = container.dom.offsetWidth / container.dom.offsetHeight;
		editor.camera.updateProjectionMatrix();

		camera.aspect = container.dom.offsetWidth / container.dom.offsetHeight;
		camera.updateProjectionMatrix();

        // To avoid the aspect ratio of the OpenSim watermark from changing
        // as the window is resized.
        editor.sceneOrthoCam.left = 0;
        editor.sceneOrthoCam.right = window.innerWidth;
        editor.sceneOrthoCam.top = window.innerHeight;
        editor.sceneOrthoCam.bottom = 0;
        editor.sceneOrthoCam.updateProjectionMatrix();

		renderer.setSize( container.dom.offsetWidth, container.dom.offsetHeight );

		render();

	} );

	signals.showGridChanged.add( function ( showGrid ) {

		grid.visible = showGrid;
		render();

	} );

	signals.hiresRender.add(function (upSample) {
		if (upSample===undefined) // From button -> 300 DPI
			renderHiRes(screenCapUpsamplingFactor);
		else // using hotkey d -> 900 DPI
			renderHiRes(screenCapUpsamplingFactor + upSample);
	});

	signals.screenCaptureScaleupChanged.add(function (newFactor) {
		screenCapUpsamplingFactor = newFactor;
	});
	//var curTime = 0;
	signals.captureFrame.add(function () {
	    recording = false;
	    render();
	    recording = true;
	    if (capturer !== undefined) {
	        capturer.capture(renderer.domElement);
	        //var newTime = performance.now();
	        //console.log('Capture delta', newTime - curTime);
	        //curTime = newTime;
        }
	});
	//
	signals.objectChanged.add(function (changedObject) {
		var test = changedObject;
		// Should handle "movable types only" based on opensimtype
		// i.e. Marker, PathPoint, eventually everything draggable
		var offsets = {
			"type": "transforms",
			uuids: [],
			positions: []
		};
		offsets.uuids.push(changedObject.uuid);
		offsets.positions.push(changedObject.position);
		sendText(JSON.stringify(offsets));
	});

	signals.moveCameraTo.add(function (targetPos, targetCenter, targetQuat) {
		viewHelper.moveCameraTo(targetPos, targetCenter, targetQuat);
		editor.sceneLight.position.copy(editor.camera.position);
	});

	var renderer = null;

	render();

	var clock = new THREE.Clock(); // only used for animations

	function updateFog( root ) {

		if ( root.fog ) {

			root.fog.color.setHex( oldFogColor );

			if ( root.fog.near !== undefined ) root.fog.near = oldFogNear;
			if ( root.fog.far !== undefined ) root.fog.far = oldFogFar;
			if ( root.fog.density !== undefined ) root.fog.density = oldFogDensity;

		}

	}
	// animations

	function animate() {

		var mixer = editor.mixer;
		var delta = clock.getDelta();

		var needsUpdate = false;

		if (mixer.stats.actions.inUse > 0) {

			mixer.update(delta);
			needsUpdate = true;

		}

		if (viewHelper.animating === true) {
			viewHelper.update(delta);
			needsUpdate = true;
			editor.sceneLight.position.copy(editor.camera.position);
		}
        /*
		if (vr.currentSession !== null) {

			needsUpdate = true;

		}
        */
		if (needsUpdate === true) render();

	}

	//

	function renderHiRes(upSample) {
		var curSize = new THREE.Vector2(0, 0);
		renderer.getSize(curSize)
		var saveWidth = curSize.width;
		var saveHeight = curSize.height;
		var widthOfScreenshot = saveWidth * upSample;
		var heightOfScreenshot = saveHeight * upSample;
		renderer.setSize(widthOfScreenshot, heightOfScreenshot);
		renderer.clear(); // clear first to keep background color
		renderer.render(scene, editor.camera);
		renderer.render(sceneOrtho, sceneOrthoCam);
		renderer.domElement.toBlob(function (blob) {
			var link = document.createElement('a');
			link.download = "opensim_snapshot.png";
			var url = URL.createObjectURL(blob);
			link.href = url;
			document.body.appendChild(link);
			link.click();
			//document.body.removeChild(link);
		});
		renderer.setSize(saveWidth, saveHeight);
		renderer.clear();
		//renderer.setClearColor(clearColor);
		renderer.render(scene, editor.camera);
		renderer.render(sceneOrtho, sceneOrthoCam);

	}

	//var clock = new THREE.Clock(); // only used for animations

	function render() {
	    if (!recording) {
	        //var t0 = performance.now();
	        sceneHelpers.updateMatrixWorld();
	        scene.updateMatrixWorld();
	        //stats.update();
	        if (renderer != null) {
	            if (viewHelper.animating===false) renderer.clear();
	            renderer.render(scene, camera);
	            renderer.render(sceneOrtho, sceneOrthoCam);
	            
				if (sceneHelpers.visible)
					renderer.render(sceneHelpers, camera);
	            
	            //if (recording) capturer.capture(renderer.domElement);
	        }
                if (editor.reportframeTime){
                    //var t1 = performance.now();
                    //editor.reportRenderTime(t1-t0);
                }
	    }
	}

	return container;

};


export { OpenSimViewport };
