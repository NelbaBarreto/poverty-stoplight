-- ============================================================
-- Migración: Demo Semáforo — tablas categoria_pregunta y pregunta
-- Ejecutar: PGPASSWORD=sg4dm1n! psql -h 10.1.50.50 -U sgadmin -d db_rag -f migrate_demo_semaforo.sql
-- ============================================================

-- ── Tabla 1: Categorías ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS categoria_pregunta (
    id           SERIAL PRIMARY KEY,
    nombre       VARCHAR(100) NOT NULL,
    nombre_corto VARCHAR(50),
    orden        INTEGER NOT NULL DEFAULT 0,
    activa       BOOLEAN NOT NULL DEFAULT true,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Tabla 2: Preguntas ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pregunta (
    id                   SERIAL PRIMARY KEY,
    categoria_id         INTEGER NOT NULL REFERENCES categoria_pregunta(id),
    numero_indicador     INTEGER NOT NULL UNIQUE,
    titulo               VARCHAR(200) NOT NULL,
    shortname            VARCHAR(100),
    lifemap_name         VARCHAR(200),
    descripcion          TEXT,
    descripcion_verde    TEXT,
    descripcion_amarillo TEXT,
    descripcion_rojo     TEXT,
    pregunta_sugerida1   TEXT,
    pregunta_sugerida2   TEXT,
    pregunta_sugerida3   TEXT,
    activa               BOOLEAN NOT NULL DEFAULT true,
    orden                INTEGER NOT NULL DEFAULT 0,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Categorías ──────────────────────────────────────────────────
INSERT INTO categoria_pregunta (nombre, nombre_corto, orden) VALUES
  ('Ingreso y Empleo',           'Ingresos',    1),
  ('Salud y Medio Ambiente',     'Salud',        2),
  ('Vivienda e Infraestructura', 'Vivienda',     3),
  ('Educación y Cultura',        'Educacion',    4),
  ('Interioridad y Motivación',  'Interioridad', 5)
ON CONFLICT DO NOTHING;

-- ── 10 Indicadores del PDF ──────────────────────────────────────
INSERT INTO pregunta (
    categoria_id, numero_indicador, titulo, shortname, lifemap_name,
    descripcion, descripcion_verde, descripcion_amarillo, descripcion_rojo,
    pregunta_sugerida1, pregunta_sugerida2, pregunta_sugerida3, orden)
VALUES

-- 1 ─ Ingresos
((SELECT id FROM categoria_pregunta WHERE nombre='Ingreso y Empleo'), 1,
 'Ingresos superiores a la línea de pobreza','Ingresos','Tenemos ingresos suficientes',
 'La Fundación Paraguaya adopta las cifras de la DGEEC basadas en la EPH 2016. Pobreza total: urbana ₲ 686.075 y rural ₲ 488.172. Pobreza extrema: urbana ₲ 262.768 y rural ₲ 239.969.',
 'Mi familia tiene ingresos que superan la línea de pobreza total.',
 'Mi familia tiene ingresos inferiores a la línea de pobreza total y superiores a la línea de pobreza extrema.',
 'Mi familia tiene ingresos inferiores a la línea de pobreza extrema.',
 '¿Los ingresos de su familia superan la línea de pobreza total?',
 '¿Cuántos miembros de su familia aportan ingresos al hogar?',
 '¿Sus ingresos cubren los gastos básicos de alimentación, salud y educación?', 1),

-- 2 ─ Ahorros
((SELECT id FROM categoria_pregunta WHERE nombre='Ingreso y Empleo'), 2,
 'Ahorros familiares','Ahorros','Tenemos ahorros',
 'Se entiende por ahorros familiares a la parte de los ingresos que una familia no gasta, guardada en caja de ahorro para necesidades futuras. Se considera la constancia del ahorro durante al menos los últimos 6 meses.',
 'Uno o más miembros de mi familia tienen ahorros hace al menos seis meses y tiene una cuenta de ahorro a su nombre.',
 'Uno o más miembros de mi familia tienen ahorros de manera informal (mantienen el dinero en casa, grupos informales de ahorro, etc.) o tienen cuenta de ahorro hace menos de seis meses.',
 'Ningún miembro de mi familia tiene ahorros.',
 '¿Algún miembro de su familia ahorra regularmente?',
 '¿Tiene cuenta de ahorro formal en una institución financiera?',
 '¿Hace cuánto tiempo practica el ahorro de manera constante?', 2),

-- 3 ─ Agua
((SELECT id FROM categoria_pregunta WHERE nombre='Salud y Medio Ambiente'), 3,
 'Acceso al agua potable','Agua','Tenemos canilla',
 'El acceso al agua potable es la posibilidad de contar constantemente con agua que puede usarse para beber o cocinar sin riesgo de enfermedades, porque no contiene sustancias peligrosas o ha sido tratada para consumo humano.',
 'Mi familia cuenta con agua potable y accede a través de una canilla dentro del terreno de la vivienda.',
 'Mi familia tiene acceso al agua potable fuera del terreno de la vivienda, a través de canilla, pozo o aljibe protegido. La distancia es menos de 30 minutos de caminata ida y vuelta.',
 'Mi familia bebe agua no potable o tiene que acarrearla desde un punto a más de 30 minutos de caminata ida y vuelta.',
 '¿Cómo accede su familia al agua potable?',
 '¿Tiene canilla dentro del terreno de su vivienda?',
 '¿El agua que consume su familia es segura para beber y cocinar?', 3),

-- 4 ─ Vacunas
((SELECT id FROM categoria_pregunta WHERE nombre='Salud y Medio Ambiente'), 4,
 'Vacunas','Vacunas','Estamos vacunados',
 'Vacunas obligatorias según el Ministerio de Salud: a) <6 meses: BCG, anti-rotavirus, IPV/OPV, pentavalente, neumococo. b) <4 años: SPR, anti-varicela, anti-hepatitis A, AA, DTP, anti-influenza, BOPV. c) >10 años: anti-VPH, TDPA.',
 'Todos los miembros de mi familia están inmunizados y al día con todas las vacunas obligatorias según el Ministerio de Salud Pública y Bienestar Social.',
 'Un miembro de mi familia no está inmunizado o al día con todas las vacunas obligatorias según el Ministerio de Salud Pública y Bienestar Social.',
 'Más de un miembro de mi familia no está inmunizado o al día con todas las vacunas obligatorias según el Ministerio de Salud Pública y Bienestar Social.',
 '¿Todos los miembros de su familia tienen las vacunas al día?',
 '¿Tiene el carnet de vacunación de todos sus hijos?',
 '¿Conoce el calendario de vacunación del Ministerio de Salud?', 4),

-- 5 ─ Cocina
((SELECT id FROM categoria_pregunta WHERE nombre='Vivienda e Infraestructura'), 5,
 'Cocina elevada y ventilada','Cocina','Tenemos cocina ventilada y elevada',
 'La familia prepara su comida sin riesgo para la salud ni peligro de incendio. Requiere: espacio cubierto, cocina elevada (80 cm o más) y ventilación suficiente. No debe usarse estiércol, carbón ni leña.',
 'Mi familia cocina en un espacio cubierto, protegido y ventilado. Cuenta con cocina elevada y no utiliza estiércol, carbón o leña.',
 'Mi familia cocina en espacio cubierto y protegido pero no ventilado, o al aire libre o en fogón. En ningún caso utiliza estiércol, carbón o leña.',
 'Mi familia cocina utilizando, principalmente, estiércol, carbón o leña.',
 '¿Cómo cocina su familia habitualmente?',
 '¿Cuenta con una cocina elevada y ventilada en su hogar?',
 '¿Utiliza leña, carbón o estiércol para cocinar?', 5),

-- 6 ─ Baños
((SELECT id FROM categoria_pregunta WHERE nombre='Vivienda e Infraestructura'), 6,
 'Baños','Baño','Tenemos baño moderno',
 'Un baño es un espacio cerrado y cubierto limpio que provee intimidad y buen sistema de evacuación (pozo ciego o desagüe cloacal).',
 'Mi familia tiene un baño con: inodoro completo (WC), cisterna, privacidad, buen sistema de evacuación, no se comparte con otra familia y se mantiene limpio.',
 'Aunque el baño no reúne todos los requisitos del nivel verde, cuenta con algún tipo de inodoro con agua, letrina con losa y buena ventilación, o inodoro orgánico. No se comparte con otra familia.',
 'Mi familia comparte su baño o letrina con otra familia o no reúne los requisitos para el nivel amarillo.',
 '¿Cuenta su hogar con un baño propio?',
 '¿Comparte el baño con otra familia?',
 '¿El baño cuenta con inodoro con agua y buen sistema de evacuación?', 6),

-- 7 ─ Electricidad
((SELECT id FROM categoria_pregunta WHERE nombre='Vivienda e Infraestructura'), 7,
 'Electricidad','Electricidad','Tenemos electricidad',
 'Acceso a la energía eléctrica de manera permanente y legal en la casa.',
 'Mi familia tiene acceso permanente y no clandestino a la electricidad.',
 'Mi familia tiene acceso a la electricidad, pero es clandestino y/o insuficiente (pasa cuatro horas o más del día sin electricidad en temporada alta).',
 'Mi familia no tiene acceso a la electricidad.',
 '¿Su hogar tiene acceso a electricidad?',
 '¿La conexión eléctrica de su casa es legal y permanente?',
 '¿Hay cortes frecuentes de electricidad en su zona?', 7),

-- 8 ─ Caminos
((SELECT id FROM categoria_pregunta WHERE nombre='Vivienda e Infraestructura'), 8,
 'Camino de acceso de todo tiempo','Caminos','Tenemos calles transitables',
 'La familia cuenta con caminos accesibles todo el tiempo para llegar al centro urbano más cercano con transporte motorizado normal, independientemente de las inclemencias del tiempo.',
 'El camino al centro urbano más cercano es asfaltado, empedrado, adoquinado o enripiado, y accesible todo el tiempo, incluso en lluvias fuertes o continuas.',
 'El camino es de tierra, o asfaltado/empedrado en mal estado y difícil de transitar en periodo de lluvias fuertes o continuas.',
 'El camino es de tierra y ante la menor inclemencia del tiempo, se vuelve difícil de transitar.',
 '¿El camino a su hogar es transitable todo el año?',
 '¿En época de lluvias puede llegar al centro urbano más cercano?',
 '¿El camino está asfaltado, empedrado o en buen estado?', 8),

-- 9 ─ Presupuesto
((SELECT id FROM categoria_pregunta WHERE nombre='Educación y Cultura'), 9,
 'Capacidad de planificar y presupuestar','Presupuesto','Tenemos presupuesto',
 'La familia proyecta y calcula por escrito su futuro económico en el corto, mediano y largo plazo, y realiza el seguimiento de sus planes.',
 'Mi familia planifica su futuro económico y elabora un presupuesto mensual escrito, que utiliza y rige la economía permanente.',
 'Mi familia planifica su futuro económico y elabora un presupuesto mensual escrito, pero no lo utiliza y no rige la economía permanentemente.',
 'Mi familia no planifica su futuro económico y no tiene un presupuesto escrito.',
 '¿Su familia planifica cómo gastar sus ingresos mensualmente?',
 '¿Elabora un presupuesto escrito que sigue regularmente?',
 '¿Tiene metas económicas a corto y mediano plazo?', 9),

-- 10 ─ Conciencia
((SELECT id FROM categoria_pregunta WHERE nombre='Interioridad y Motivación'), 10,
 'Conciencia de sus necesidades','Conciencia necesidades','Tenemos nuestro mapa de vida',
 'La conciencia de sus necesidades se refiere a que los miembros de una familia comprenden cuáles son sus necesidades, más allá de las básicas, y tienen metas a mediano y largo plazo.',
 'Mi familia sabe que puede mejorar su situación actual. Tiene metas alcanzables en el mediano (seis meses) y largo plazo (un año o más). Realiza acciones continuamente para lograrlas.',
 'Mi familia piensa que puede mejorar su situación actual, pero no tiene metas a mediano (seis meses) ni a largo plazo (un año o más).',
 'Mi familia piensa que no puede mejorar su situación actual, no quiere hacerlo, o no la comprende. En consecuencia, no tiene metas de corto ni largo plazo.',
 '¿Su familia tiene metas a mediano y largo plazo?',
 '¿Sabe qué necesita mejorar para tener mejor calidad de vida?',
 '¿Realiza acciones concretas para lograr sus metas?', 10)

ON CONFLICT (numero_indicador) DO NOTHING;

-- Verificación
SELECT cp.nombre AS categoria, p.numero_indicador, p.shortname
FROM pregunta p
JOIN categoria_pregunta cp ON p.categoria_id = cp.id
ORDER BY p.numero_indicador;
