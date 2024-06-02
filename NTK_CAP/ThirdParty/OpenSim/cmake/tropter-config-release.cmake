#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tropter" for configuration "Release"
set_property(TARGET tropter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tropter PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/sdk/lib/tropter.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/tropter.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS tropter )
list(APPEND _IMPORT_CHECK_FILES_FOR_tropter "${_IMPORT_PREFIX}/sdk/lib/tropter.lib" "${_IMPORT_PREFIX}/bin/tropter.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
