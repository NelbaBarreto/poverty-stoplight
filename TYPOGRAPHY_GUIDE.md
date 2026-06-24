# Sistema de Tipografía Unificado - Guía de Uso

## Descripción General

Se ha implementado un **sistema de escala tipográfica consistente** basado en variables CSS. Esto asegura que todos los textos en el sitio tengan tamaños y espaciados uniformes.

---

## Variables CSS Disponibles

### Tamaños de Fuente
```css
--font-size-xs:   0.75rem   (12px)
--font-size-sm:   0.875rem  (14px)
--font-size-base: 1rem      (16px)     ← DEFAULT
--font-size-lg:   1.125rem  (18px)
--font-size-xl:   1.25rem   (20px)
--font-size-2xl:  1.5rem    (24px)
--font-size-3xl:  1.875rem  (30px)
--font-size-4xl:  2.25rem   (36px)
--font-size-5xl:  3rem      (48px)
--font-size-6xl:  3.75rem   (60px)
```

### Alturas de Línea
```css
--line-height-tight:   1.2  ← Headings
--line-height-normal:  1.5  ← Párrafos (DEFAULT)
--line-height-relaxed: 1.75 ← Textos largos
--line-height-loose:   2    ← Muy espaciado
```

### Pesos de Fuente
```css
--font-weight-light:     300
--font-weight-normal:    400 (DEFAULT)
--font-weight-semibold:  600 ← Headings
--font-weight-bold:      700 ← Títulos
```

---

## Elementos HTML Predefinidos

Estos elementos **ya tienen estilos aplicados automáticamente**:

### Headings
- `<h1>` → 48px, bold, línea 1.2
- `<h2>` → 36px, bold, línea 1.2
- `<h3>` → 30px, semibold, línea 1.2
- `<h4>` → 24px, semibold, línea 1.5
- `<h5>` → 20px, semibold, línea 1.5
- `<h6>` → 18px, semibold, línea 1.5

### Párrafos
- `<p>` → 16px, normal, línea 1.5
- `<small>` / `.text-sm` → 14px

### Otros
- `<label>` → 16px, semibold
- `<code>` → 14px, monospace
- `<blockquote>` → 18px, italic

---

## Clases de Utilidad para Personalización

### Tamaños
```html
<p class="text-xs">Muy pequeño</p>
<p class="text-sm">Pequeño</p>
<p class="text-lg">Grande</p>
<p class="text-xl">Muy grande</p>
<p class="text-2xl">Extra grande</p>
<p class="text-3xl">Gigante</p>
```

### Pesos
```html
<span class="text-light">Texto delgado</span>
<span class="text-semibold">Texto semi-grueso</span>
<span class="text-bold">Texto grueso</span>
```

### Alturas de Línea
```html
<p class="line-height-tight">Línea apretada</p>
<p class="line-height-normal">Línea normal</p>
<p class="line-height-relaxed">Línea relajada</p>
<p class="line-height-loose">Línea muy espaciada</p>
```

---

## Ejemplos Prácticos

### ✅ CORRECTO - Uso de variables
```html
<!-- En un componente Angular -->
<div class="content">
  <h2>Título Principal</h2>
  <p>Este párrafo usa automáticamente 16px y espaciado consistente.</p>
  <h3>Subtítulo</h3>
  <p class="text-sm">Este texto es más pequeño para detalles.</p>
</div>
```

El CSS va automáticamente:
```css
h2 { font-size: var(--font-size-4xl); }
p { font-size: var(--font-size-base); }
```

### ✅ CORRECTO - Personalización con clases
```html
<h1 class="text-3xl">Título personalizado</h1>
<p class="text-lg line-height-loose">
  Párrafo con líneas más espaciadas.
</p>
```

### ❌ INCORRECTO - No usar px directamente
```css
/* NO HACER ESTO */
p { font-size: 18px; }
h2 { font-size: 30px; }

/* HACER ESTO EN LUGAR */
p { font-size: var(--font-size-lg); }
h2 { font-size: var(--font-size-3xl); }
```

---

## En Componentes TypeScript/Angular

Si necesitas estilos especiales en un componente, usa las variables:

```typescript
// En component.component.scss
.custom-text {
  font-size: var(--font-size-lg);
  line-height: var(--line-height-relaxed);
  font-weight: var(--font-weight-semibold);
}
```

---

## Responsive

Los tamaños se ajustan automáticamente en **pantallas pequeñas (≤768px)**:

- `--font-size-5xl`: 48px → 36px
- `--font-size-4xl`: 36px → 30px
- `--font-size-3xl`: 30px → 24px

**No necesitas hacer media queries especiales para tipografía** — ¡ya está incluido!

---

## Verificación

Para verificar que todo está funcionando:

1. Abre el navegador → F12 (Developer Tools)
2. Ve a la pestaña "Console"
3. Ejecuta: `getComputedStyle(document.querySelector('h2')).fontSize`
4. Deberías ver algo como `"36px"` (2.25rem × 16px base)

---

## Actualización de Componentes Existentes

Si encuentras componentes con estilos de tipografía inconsistentes:

1. **Antes**: `style="font-size: 20px;"`
2. **Después**: `class="text-xl"` o `style="font-size: var(--font-size-xl);"`

Ejemplo:
```html
<!-- ANTES -->
<h3 style="font-size: 20px;">Título inconsistente</h3>

<!-- DESPUÉS -->
<h3 class="text-xl">Título consistente</h3>
<!-- O simplemente usar <h3> sin modificar -->
```

---

## Soporte

- Archivo CSS: `src/assets/css/typography.css`
- Importado en: `src/index.html`
- Variables definidas en: `:root` del typography.css
