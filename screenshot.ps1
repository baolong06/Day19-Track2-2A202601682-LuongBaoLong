Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$screens = [System.Windows.Forms.Screen]::AllScreens
$bounds = $screens[0].Bounds
$bitmap = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$graphics.Dispose()
$path = "e:\AI_thucchien\lab\Day19-Track2-2A202601682-LuongBaoLong\submission\screenshots\desktop.png"
$bitmap.Save($path)
$bitmap.Dispose()
Write-Output "Screenshot saved to $path"
