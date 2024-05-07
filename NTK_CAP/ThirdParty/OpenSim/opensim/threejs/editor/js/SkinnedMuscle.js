
/**
 * @author Ayman Habib
 */
SkinnedMuscle = function(geom, material, points, actives) {
    // Create bones for uuids in geometryPath
    this.pathpoints = points;
    this.pathpointObjects = [];
    this.actives = actives;
    var bones = [];
    this.firstPointMaterial = undefined;
    // When off only caps are shown, when on, user defined points are shown (for editing/picking)
    this.showInnerPathPoints = false;

    for (var i=0; i< 2*points.length-2; i++) {
        var bone = new THREE.Bone();
        bone.pos = [0, 0, 0];
        bone.rotq = [0, 0, 0, 1];
        bone.ppt = this.pathpoints[Math.floor(i/2)];
        bones.push(bone);
    }

    var numVerticesPerLevel = geom.attributes.position.count / (2*points.length-2);
    const skinIndices = [];
    const skinWeights = [];
    const position = geom.attributes.position;
    for ( var i = 0; i < position.count; i++ ) {
        var skinIndex = Math.floor(i / numVerticesPerLevel);
        var pptIndex = Math.floor((skinIndex+1)/2);
        //var activePoint = this.actives[pptIndex];
        skinIndices.push(skinIndex, 0, 0, 0);
        // Will always use weight of 1, 0, 0, 0 to interpolate pathpoints
        // Changing weights will make the paths smooth but doesn't interpolate points'
        skinWeights.push(1.0, 0, 0, 0 );
    }
    //geom.dynamic = true;
    geom.setAttribute( 'skinIndex', new THREE.Uint16BufferAttribute( skinIndices, 4 ) );
    geom.setAttribute( 'skinWeight', new THREE.Float32BufferAttribute( skinWeights, 4 ) );
    material.skinning = true;
    THREE.SkinnedMesh.call( this, geom, material );
    const skeleton = new THREE.Skeleton( bones );
    for (var j = 0; j < bones.length; j++)
        this.add( skeleton.bones[ j ]);
    //this.add( skeleton.bones[ 1 ]);
    this.bind( skeleton );
    this.frustumCulled = false;
    this.userData = 'NonEditable';
};

SkinnedMuscle.prototype = Object.assign(Object.create( THREE.SkinnedMesh.prototype ), {
    constructor: SkinnedMuscle,

    setColor: function (newColor) {
        this.material.color.setHex(newColor).convertSRGBToLinear();
        if (this.firstPointMaterial !== undefined)
            this.firstPointMaterial.color.setHex(newColor).convertSRGBToLinear();
    },
    updateMatrixWorld: function ( force ) {
        // if has pathpoints attribute then it's a muscle
        // Cycle through pathpoints, update their matrixworld
        // then set the position of the Bones from that
        if (this.skeleton === undefined)
        return;
        const bones = this.skeleton.bones;
        //console.warn("Num bones in updateMatrixWorld: " + bones.length);

        if (this.pathpointObjects.length != this.pathpoints.length){
        let b = 0;
        for ( var p=0; p < this.pathpoints.length; p++) {
            let pptObject1 = editor.objectByUuid(this.pathpoints[p]);
            if (this.firstPointMaterial === undefined && pptObject1 !== undefined && p === 0)
                this.firstPointMaterial = pptObject1.material;
            let pptObject2 = editor.objectByUuid(this.pathpoints[p+1]);

            if (pptObject1 !== undefined) {
            // add every pathpoint to the list of PathPoint objects
                this.pathpointObjects.push(pptObject1);

                if (pptObject2 !== undefined) {
                // define the two bones of a segement of the path together
                    bones[b].geometry = pptObject1.geometry;
                    bones[++b].geometry = pptObject2.geometry;
                }
                b++;
            }
        }
        }
        // Compute reverse transform from Ground to Scene (usually this's inverse translation)
        // This is necessary since the blending to compute vertices adds offset twice
        if (this.parent === null) return; // construction
        var mat = new THREE.Matrix4().copy(this.parent.matrixWorld).invert();
        var vec = new THREE.Vector3().setFromMatrixPosition(mat);

        // Variables for the two points of a given path segement, the axis to
        // be rotated (from) and the vector between them (to)
        var pt1 = new THREE.Vector3();
        var pt2 = new THREE.Vector3();
        var vFrom = new THREE.Vector3(0, -1, 0);
        var vTo = new THREE.Vector3();

        // cycle through each segement defined by two PathPoints, pt1 and pt2
        // and alignge the bones (caps of each segment) to be alinged with
        // the vector connecting them.
        var b = 0; // bone (of SkinnedMuscle) index
        for (var p = 0; p < this.pathpoints.length-1; p++) {
        var thisPathpointObject = this.pathpointObjects[p];
        var nextPathpointObject = this.pathpointObjects[p+1];

        if(thisPathpointObject !== undefined) {
            pt1.setFromMatrixPosition(thisPathpointObject.matrixWorld);
            pt2.setFromMatrixPosition(nextPathpointObject.matrixWorld);

            vTo = pt2.clone();
            vTo.sub(pt1).normalize();

            // bones are positioned on the pathpoints
            bones[b].position.setFromMatrixPosition(thisPathpointObject.matrixWorld);
            bones[b].position.add(vec);
            // the orientation of the bone is updated to have its Y-axis pointed
            // back along the vector from pt1 to pt2
            bones[b].quaternion.setFromUnitVectors(vFrom, vTo);

            bones[++b].position.setFromMatrixPosition(nextPathpointObject.matrixWorld);
            // the orientation of the bone is updated to have its Y-axis pointed
            // back along the vector from pt1 to pt2
            bones[b].position.add(vec);
            bones[b++].quaternion.setFromUnitVectors(vFrom, vTo);
        }
    }
    this.skeleton.update();
    THREE.SkinnedMesh.prototype.updateMatrixWorld.call( this, true );

    },
    setVisible: function ( newValue) {
        this.visible = newValue;
        // Now repeat for the inner pathpoints under this muscle
        for (var p = 0; p < this.pathpoints.length; p++) {
            if (this.pathpointObjects[p].opensimType!=="ComputedPathPoint")
                this.pathpointObjects[p].visible = newValue;
        }
    },
    togglePathPoints: function (newValue) {
        for (var p = 0; p < this.pathpoints.length; p++) {
            if (this.actives[p])
                this.pathpointObjects[p].visible = newValue;
        }
        
    }
    
});
