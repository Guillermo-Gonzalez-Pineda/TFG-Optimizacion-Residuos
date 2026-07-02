# cmake_minimum_required: 3.16 = base amplia y estable. file(GLOB CONFIGURE_DEPENDS)
# necesita >= 3.12, así que 3.16 lo cubre de sobra.
cmake_minimum_required(VERSION 3.16)
project(metaheuristica LANGUAGES CXX)

# --- Estándar: C++20 EXIGIDO -------------------------------------------------
# REQUIRED ON convierte C++20 en requisito duro: si el compilador no lo soporta,
# la CONFIGURACIÓN falla ruidosamente (en vez de bajar en silencio a C++17 y
# compilar un binario distinto o con errores crípticos). Fallar pronto y claro.
# EXTENSIONS OFF fuerza -std=c++20 (estándar estricto), no -std=gnu++20 (dialecto).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Build type por defecto = Release ----------------------------------------
# Si el usuario no pasa -DCMAKE_BUILD_TYPE, asumimos Release (binario de
# producción). Sin esto, un build sin tipo no aplicaría ni -O2 ni el oráculo.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release | Debug" FORCE)
endif()

# --- Flags por build type: el ORÁCULO lo activa SOLO Debug -------------------
# Ambos a -O2. Debug NO es "depuración con gdb" sino "modo ORÁCULO": el oráculo
# reconstruye el estado desde cero cada iteración → a -O0 va ~7x más lento; a -O2
# es IEEE-estable (mismo veredicto y misma trayectoria). -g no aporta al oráculo
# (aborta e imprime diagnóstico igual), por eso se omite. Decisión consciente.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG   "-O2 -DORACLE_CHECK")

# --- Fuentes -----------------------------------------------------------------
# GLOB + CONFIGURE_DEPENDS recoge todos los src/*.cpp (incluido oracle.cpp) y
# re-detecta ficheros nuevos al compilar. Es una decisión de COMODIDAD para
# nuestra escala (8 ficheros). Alternativa purista (recomendada en proyectos
# grandes): listar las fuentes a mano, para que CMake sepa exactamente de qué
# depende y no dependa de re-globear.
file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(metaheuristica ${SOURCES})

# El compilador NO se fija aquí (portabilidad): el fichero fija el "qué" (C++20,
# flags, fuentes); el "quién" (g++-13, g++ 15.2, clang++) se elige al configurar
# por autodetección o con -DCMAKE_CXX_COMPILER=...
target_include_directories(metaheuristica PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
