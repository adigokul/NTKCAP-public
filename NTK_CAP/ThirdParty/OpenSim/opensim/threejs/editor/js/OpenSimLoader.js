import * as THREE from '../../build/three.module.js';
import * as Geometries from '../../src/geometries/Geometries.js';

class OpenSimLoader extends THREE.ObjectLoader {

	constructor ( manager ) {
		super( manager);
	}
	// Absent sane extension mechanism this is a replica of ObjectLoader.parseGeometries
	// Modified to handle types 
	parseGeometries( json, shapes ) {

		const geometries = {};
		let geometryShapes;

		if ( json !== undefined ) {

			const bufferGeometryLoader = new THREE.BufferGeometryLoader();

			for ( let i = 0, l = json.length; i < l; i ++ ) {

				let geometry;
				const data = json[ i ];

				switch ( data.type ) {

					case 'PlaneGeometry':
					case 'PlaneBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.width,
							data.height,
							data.widthSegments,
							data.heightSegments
						);

						break;

					case 'BoxGeometry':
					case 'BoxBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.width,
							data.height,
							data.depth,
							data.widthSegments,
							data.heightSegments,
							data.depthSegments
						);

						break;

					case 'CircleGeometry':
					case 'CircleBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radius,
							data.segments,
							data.thetaStart,
							data.thetaLength
						);

						break;

					case 'CylinderGeometry':
					case 'CylinderBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radiusTop,
							data.radiusBottom,
							data.height,
							data.radialSegments,
							data.heightSegments,
							data.openEnded,
							data.thetaStart,
							data.thetaLength
						);

						break;

					case 'ConeGeometry':
					case 'ConeBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radius,
							data.height,
							data.radialSegments,
							data.heightSegments,
							data.openEnded,
							data.thetaStart,
							data.thetaLength
						);

						break;

					case 'SphereGeometry':
					case 'SphereBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radius,
							data.widthSegments,
							data.heightSegments,
							data.phiStart,
							data.phiLength,
							data.thetaStart,
							data.thetaLength
						);

						break;

					case 'DodecahedronGeometry':
					case 'DodecahedronBufferGeometry':
					case 'IcosahedronGeometry':
					case 'IcosahedronBufferGeometry':
					case 'OctahedronGeometry':
					case 'OctahedronBufferGeometry':
					case 'TetrahedronGeometry':
					case 'TetrahedronBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radius,
							data.detail
						);

						break;

					case 'RingGeometry':
					case 'RingBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.innerRadius,
							data.outerRadius,
							data.thetaSegments,
							data.phiSegments,
							data.thetaStart,
							data.thetaLength
						);

						break;

					case 'TorusGeometry':
					case 'TorusBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radius,
							data.tube,
							data.radialSegments,
							data.tubularSegments,
							data.arc
						);

						break;

					case 'TorusKnotGeometry':
					case 'TorusKnotBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.radius,
							data.tube,
							data.tubularSegments,
							data.radialSegments,
							data.p,
							data.q
						);

						break;

					case 'TubeGeometry':
					case 'TubeBufferGeometry':

						// This only works for built-in curves (e.g. CatmullRomCurve3).
						// User defined curves or instances of CurvePath will not be deserialized.
						geometry = new Geometries[ data.type ](
							new Curves[ data.path.type ]().fromJSON( data.path ),
							data.tubularSegments,
							data.radius,
							data.radialSegments,
							data.closed
						);

						break;

					case 'LatheGeometry':
					case 'LatheBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.points,
							data.segments,
							data.phiStart,
							data.phiLength
						);

						break;

					case 'PolyhedronGeometry':
					case 'PolyhedronBufferGeometry':

						geometry = new Geometries[ data.type ](
							data.vertices,
							data.indices,
							data.radius,
							data.details
						);

						break;

					case 'ShapeGeometry':
					case 'ShapeBufferGeometry':

						geometryShapes = [];

						for ( let j = 0, jl = data.shapes.length; j < jl; j ++ ) {

							const shape = shapes[ data.shapes[ j ] ];

							geometryShapes.push( shape );

						}

						geometry = new Geometries[ data.type ](
							geometryShapes,
							data.curveSegments
						);

						break;


					case 'ExtrudeGeometry':
					case 'ExtrudeBufferGeometry':

						geometryShapes = [];

						for ( let j = 0, jl = data.shapes.length; j < jl; j ++ ) {

							const shape = shapes[ data.shapes[ j ] ];

							geometryShapes.push( shape );

						}

						const extrudePath = data.options.extrudePath;

						if ( extrudePath !== undefined ) {

							data.options.extrudePath = new Curves[ extrudePath.type ]().fromJSON( extrudePath );

						}

						geometry = new Geometries[ data.type ](
							geometryShapes,
							data.options
						);

						break;

					case 'BufferGeometry':
					case 'InstancedBufferGeometry':

						geometry = bufferGeometryLoader.parse( data );

						break;
					// OPENSIM Specific Types
					case 'PathGeometry':
						//CylinderGeometry(radiusTop, radiusBottom, height, radiusSegments, heightSegments, openEnded, thetaStart, thetaLength)
						// NOTE: number of heightSegments must be equal to the number of bones in the SkinnedMuscle
						geometry = new THREE.CylinderGeometry(data.radius, data.radius, 0.1, 8, 2*data.segments-1, true);
		
						break;
					case 'Geometry':

						console.error( 'THREE.OpenSimLoader: Loading "Geometry" is not supported anymore.' );

						break;

					default:

						console.warn( 'THREE.OpenSimLoader: Unsupported geometry type "' + data.type + '"' );

						continue;

				}

				geometry.uuid = data.uuid;

				if ( data.name !== undefined ) geometry.name = data.name;
				if ( geometry.isBufferGeometry === true && data.userData !== undefined ) geometry.userData = data.userData;

				geometries[ data.uuid ] = geometry;

			}

		}

		return geometries;

	}
	parseObject( data, geometries, materials, animations ) {

		let object;

		function getGeometry( name ) {

			if ( geometries[ name ] === undefined ) {

				console.warn( 'THREE.ObjectLoader: Undefined geometry', name );

			}

			return geometries[ name ];

		}

		function getMaterial( name ) {

			if ( name === undefined ) return undefined;

			if ( Array.isArray( name ) ) {

				const array = [];

				for ( let i = 0, l = name.length; i < l; i ++ ) {

					const uuid = name[ i ];

					if ( materials[ uuid ] === undefined ) {

						console.warn( 'THREE.ObjectLoader: Undefined material', uuid );

					}

					array.push( materials[ uuid ] );

				}

				return array;

			}

			if ( materials[ name ] === undefined ) {

				console.warn( 'THREE.ObjectLoader: Undefined material', name );

			}

			return materials[ name ];

		}

		let geometry, material;

		switch ( data.type ) {

			case 'Scene':

				object = new Scene();

				if ( data.background !== undefined ) {

					if ( Number.isInteger( data.background ) ) {

						object.background = new Color( data.background );

					}

				}

				if ( data.fog !== undefined ) {

					if ( data.fog.type === 'Fog' ) {

						object.fog = new Fog( data.fog.color, data.fog.near, data.fog.far );

					} else if ( data.fog.type === 'FogExp2' ) {

						object.fog = new FogExp2( data.fog.color, data.fog.density );

					}

				}

				break;

			case 'PerspectiveCamera':

				object = new PerspectiveCamera( data.fov, data.aspect, data.near, data.far );

				if ( data.focus !== undefined ) object.focus = data.focus;
				if ( data.zoom !== undefined ) object.zoom = data.zoom;
				if ( data.filmGauge !== undefined ) object.filmGauge = data.filmGauge;
				if ( data.filmOffset !== undefined ) object.filmOffset = data.filmOffset;
				if ( data.view !== undefined ) object.view = Object.assign( {}, data.view );

				break;

			case 'OrthographicCamera':

				object = new OrthographicCamera( data.left, data.right, data.top, data.bottom, data.near, data.far );

				if ( data.zoom !== undefined ) object.zoom = data.zoom;
				if ( data.view !== undefined ) object.view = Object.assign( {}, data.view );

				break;

			case 'AmbientLight':

				object = new AmbientLight( data.color, data.intensity );

				break;

			case 'DirectionalLight':

				object = new DirectionalLight( data.color, data.intensity );

				break;

			case 'PointLight':

				object = new PointLight( data.color, data.intensity, data.distance, data.decay );

				break;

			case 'RectAreaLight':

				object = new RectAreaLight( data.color, data.intensity, data.width, data.height );

				break;

			case 'SpotLight':

				object = new SpotLight( data.color, data.intensity, data.distance, data.angle, data.penumbra, data.decay );

				break;

			case 'HemisphereLight':

				object = new HemisphereLight( data.color, data.groundColor, data.intensity );

				break;

			case 'LightProbe':

				object = new LightProbe().fromJSON( data );

				break;

			case 'SkinnedMesh':

				geometry = getGeometry( data.geometry );
			 	material = getMaterial( data.material );

				object = new SkinnedMesh( geometry, material );

				if ( data.bindMode !== undefined ) object.bindMode = data.bindMode;
				if ( data.bindMatrix !== undefined ) object.bindMatrix.fromArray( data.bindMatrix );
				if ( data.skeleton !== undefined ) object.skeleton = data.skeleton;

				break;

			case 'Mesh':

				geometry = getGeometry( data.geometry );
				material = getMaterial( data.material );
				geometry.computeVertexNormals();
				object = new THREE.Mesh( geometry, material );
				break;

			case 'InstancedMesh':

				geometry = getGeometry( data.geometry );
				material = getMaterial( data.material );
				const count = data.count;
				const instanceMatrix = data.instanceMatrix;

				object = new InstancedMesh( geometry, material, count );
				object.instanceMatrix = new BufferAttribute( new Float32Array( instanceMatrix.array ), 16 );

				break;

			case 'LOD':

				object = new LOD();

				break;

			case 'Line':

				object = new Line( getGeometry( data.geometry ), getMaterial( data.material ) );

				break;

			case 'LineLoop':

				object = new LineLoop( getGeometry( data.geometry ), getMaterial( data.material ) );

				break;

			case 'LineSegments':

				object = new LineSegments( getGeometry( data.geometry ), getMaterial( data.material ) );

				break;

			case 'PointCloud':
			case 'Points':

				object = new Points( getGeometry( data.geometry ), getMaterial( data.material ) );

				break;

			case 'Sprite':

				object = new Sprite( getMaterial( data.material ) );

				break;

			case 'Group':

				object = new THREE.Group();

				break;

			case 'Bone':

				object = new Bone();

				break;
			// OPENSIM Types
			case 'Frame':

				object = new THREE.AxesHelper(data.size);

				break;
			case 'GeometryPath':
                object = new SkinnedMuscle(getGeometry(data.geometry),  getMaterial( data.material ), data.points, data.active);

                break;

            case 'Arrow':
                object = new THREE.ArrowHelper(data.dir, data.origin, 1000, data.color);

                break;

			default:

				object = new THREE.Object3D();

		}
		if (object === undefined )
			return object;

		object.uuid = data.uuid;

		if ( data.name !== undefined ) object.name = data.name;

		if ( data.matrix !== undefined ) {

			object.matrix.fromArray( data.matrix );

			if ( data.matrixAutoUpdate !== undefined ) object.matrixAutoUpdate = data.matrixAutoUpdate;
			if ( object.matrixAutoUpdate ) object.matrix.decompose( object.position, object.quaternion, object.scale );

		} else {

			if ( data.position !== undefined ) object.position.fromArray( data.position );
			if ( data.rotation !== undefined ) object.rotation.fromArray( data.rotation );
			if ( data.quaternion !== undefined ) object.quaternion.fromArray( data.quaternion );
			if ( data.scale !== undefined ) object.scale.fromArray( data.scale );

		}

		if ( data.castShadow !== undefined ) object.castShadow = data.castShadow;
		if ( data.receiveShadow !== undefined ) object.receiveShadow = data.receiveShadow;

		if ( data.shadow ) {

			if ( data.shadow.bias !== undefined ) object.shadow.bias = data.shadow.bias;
			if ( data.shadow.normalBias !== undefined ) object.shadow.normalBias = data.shadow.normalBias;
			if ( data.shadow.radius !== undefined ) object.shadow.radius = data.shadow.radius;
			if ( data.shadow.mapSize !== undefined ) object.shadow.mapSize.fromArray( data.shadow.mapSize );
			if ( data.shadow.camera !== undefined ) object.shadow.camera = this.parseObject( data.shadow.camera );

		}

		if ( data.visible !== undefined ) object.visible = data.visible;
		if ( data.frustumCulled !== undefined ) object.frustumCulled = data.frustumCulled;
		if ( data.renderOrder !== undefined ) object.renderOrder = data.renderOrder;
		if ( data.userData !== undefined ) object.userData = data.userData;
		if ( data.layers !== undefined ) object.layers.mask = data.layers;
		// OPENSIM Specific Handling to make object nonEditable
		if ( object.userData === 'NonEditable') { // Propagate to children if any
			for ( var child in object.children ) {
				object.children[child].userData = 'NonEditable';
			}
		}
		if ( data.opensimType !== undefined ) {
				object.opensimType = data.opensimType;
		}
		// end OPENSIM Specific Handling
		if ( data.children !== undefined ) {

			const children = data.children;

			for ( let i = 0; i < children.length; i ++ ) {
				var nextObj = this.parseObject( children[ i ], geometries, materials, animations );
				if (nextObj !== undefined)
					object.add( nextObj );

			}

		}

		if ( data.animations !== undefined ) {

			const objectAnimations = data.animations;

			for ( let i = 0; i < objectAnimations.length; i ++ ) {

				const uuid = objectAnimations[ i ];

				object.animations.push( animations[ uuid ] );

			}

		}

		if ( data.type === 'LOD' ) {

			if ( data.autoUpdate !== undefined ) object.autoUpdate = data.autoUpdate;

			const levels = data.levels;

			for ( let l = 0; l < levels.length; l ++ ) {

				const level = levels[ l ];
				const child = object.getObjectByProperty( 'uuid', level.object );

				if ( child !== undefined ) {

					object.addLevel( child, level.distance );

				}

			}

		}

		return object;

	}

	handleJSON( data ) {

		if ( data.metadata === undefined ) { // 2.0

			data.metadata = { type: 'Geometry' };

		}

		if ( data.metadata.type === undefined ) { // 3.0

			data.metadata.type = 'Geometry';

		}

		if ( data.metadata.formatVersion !== undefined ) {

			data.metadata.version = data.metadata.formatVersion;

		}

		switch ( data.metadata.type.toLowerCase() ) {

			case 'buffergeometry':

				var loader = new THREE.BufferGeometryLoader();
				var result = loader.parse( data );

				var mesh = new THREE.Mesh( result );

				editor.execute( new AddObjectCommand( editor, mesh ) );

				break;

			case 'geometry':

				console.error( 'Loader: "Geometry" is no longer supported.' );

				break;

			case 'object':

				var loader = new THREE.ObjectLoader();
				loader.setResourcePath( scope.texturePath );

				loader.parse( data, function ( result ) {

					if ( result.isScene ) {

						editor.execute( new SetSceneCommand( editor, result ) );

					} else {

						editor.execute( new AddObjectCommand( editor, result ) );

					}

				} );

				break;

			case 'app':

				editor.fromJSON( data );

				break;

				}

	}

}

export { OpenSimLoader };
