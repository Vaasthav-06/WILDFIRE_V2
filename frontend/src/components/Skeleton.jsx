export default function Skeleton({ height = 20, width = '100%', style = {} }) {
  return (
    <div
      style={{
        height, width,
        background: 'linear-gradient(90deg, #1a2420 25%, #24312b 50%, #1a2420 75%)',
        backgroundSize: '200% 100%',
        animation: 'wildfire-skeleton-pulse 1.4s ease-in-out infinite',
        borderRadius: '6px',
        ...style,
      }}
    />
  );
}
