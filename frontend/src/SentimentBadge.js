const SentimentBadge = ({ sentiments }) => {
  const getColorWithIntensity = (baseColor, value) => {
    const intensityFactor = 0.5 + value * 0.5;
    return `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${intensityFactor})`;
  };

  const getFontColor = (backgroundColor) => {
    const brightness =
      (backgroundColor.r * 299 +
        backgroundColor.g * 587 +
        backgroundColor.b * 114) /
      1000;
    return brightness > 125 ? "black" : "white";
  };

  const totalValue = sentiments.reduce((sum, { value }) => sum + value, 0);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        height: "35px",
        borderRadius: "20px",
        overflow: "hidden",
        maxWidth: "60%",
        marginLeft: "0",
        alignSelf: "flex-start",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      }}
    >
      {sentiments.map(({ type, value }, index) => {
        if (value === 0) return null;
        const baseColors = {
          positive: { r: 0, g: 128, b: 0 },
          neutral: { r: 128, g: 128, b: 128 },
          negative: { r: 255, g: 0, b: 0 },
        };
        const widthPercent = (value / totalValue) * 100;
        const backgroundColor = getColorWithIntensity(baseColors[type], value);
        const fontColor = getFontColor(backgroundColor);

        const tooltipText = `${
          type.charAt(0).toUpperCase() + type.slice(1)
        }: ${(value * 100).toFixed(1)}%`;

        return (
          <div
            key={index}
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              width: `${widthPercent}%`,
              backgroundColor: backgroundColor,
              color: fontColor,
              fontSize: "1rem",
              transition: "width 0.3s ease",
            }}
            title={tooltipText}
          >
            <span style={{ padding: "0 5px" }}>
              {(value * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default SentimentBadge;
